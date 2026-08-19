#include "stream_encoder.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>
#ifdef WITH_GPU
#include <cuda_runtime.h>
#include "csrc/vision/processors/cuda/bgr_to_nv12.cuh"
#endif

StreamEncoder::StreamEncoder(const EncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.bitrate_kbps <= 0) cfg_.bitrate_kbps = 4000;
    if (cfg_.gop <= 0) cfg_.gop = 25 * 2; // fallback，open 时按实际 fps 修正
}

StreamEncoder::~StreamEncoder() {
    stop_async();
    close();
}

bool StreamEncoder::open(const std::string& output_url, int width, int height, int source_fps) {
    if (open_permanently_failed_.load()) return false;
    close();
    width_ = width;
    height_ = height;
    // 使用源帧率：cfg.fps=0 时自动匹配源流帧率
    if (cfg_.fps <= 0) {
        cfg_.fps = source_fps;
    }
    if (cfg_.fps <= 0) cfg_.fps = 25; // 兜底
    if (cfg_.gop <= 0) cfg_.gop = cfg_.fps * 2;
    if (!init_encoder(width, height)) return false;
    if (!open_output(output_url)) {
        // 首次推流失败即标记永久失败（地址被占用或不可用）
        open_permanently_failed_ = true;
        last_error_ = "推流地址被占用或不可用: " + output_url;
        std::cerr << "[Encoder] PERMANENT FAILURE: " << last_error_ << std::endl;
        return false;
    }
    open_retries_ = 0;
    return true;
}

void StreamEncoder::close() {
    // 只有成功写入过 header 才做完整刷写和 finalize，否则直接释放资源
    if (enc_ctx_ && header_written_) {
        // 发送 nullptr 触发编码器刷新
        avcodec_send_frame(enc_ctx_, nullptr);
        AVPacket* pkt = av_packet_alloc();
        while (avcodec_receive_packet(enc_ctx_, pkt) == 0) {
            if (stream_) {
                av_packet_rescale_ts(pkt, enc_ctx_->time_base, stream_->time_base);
                pkt->stream_index = stream_->index;
            }
            if (fmt_ctx_) av_interleaved_write_frame(fmt_ctx_, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }
    if (sws_ctx_) sws_freeContext(sws_ctx_);
#ifdef WITH_GPU
    if (gpu_nv12_buf_) cudaFree(gpu_nv12_buf_);
    gpu_nv12_buf_ = nullptr;
    gpu_nv12_capacity_ = 0;
    if (gpu_d2h_stream_) { cudaStreamDestroy(gpu_d2h_stream_); gpu_d2h_stream_ = nullptr; }
#endif
    if (enc_frame_) av_frame_free(&enc_frame_);
    if (enc_pkt_) av_packet_free(&enc_pkt_);
    if (fmt_ctx_) {
        if (header_written_) av_write_trailer(fmt_ctx_);
        if (fmt_ctx_->pb && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
    }
    if (enc_ctx_) avcodec_free_context(&enc_ctx_);
    sws_ctx_ = nullptr;
    sws_src_stride_ = 0;
    fmt_ctx_ = nullptr;
    enc_ctx_ = nullptr;
    stream_ = nullptr;
    opened_ = false;
    frame_pts_ = 0;
    encode_frame_count_ = 0;
    header_written_ = false;
    open_retries_ = 0;
    open_permanently_failed_ = false;
    last_error_.clear();
}

bool StreamEncoder::init_encoder(int width, int height) {
    // 编码器候选队列：cfg.codec=auto 时优先 h264_nvenc → libx264 → 默认 H264。
    // NVENC 打开失败（并发会话上限等硬件限制）自动回退软编，保证各输出路都能编码。
    std::vector<const AVCodec*> candidates;
    auto add_name = [&](const char* n) {
        if (n && *n) {
            if (auto* c = avcodec_find_encoder_by_name(n)) candidates.push_back(c);
        }
    };
    if (cfg_.codec == "libx264" || cfg_.codec == "x264") {
        add_name("libx264");
    } else if (cfg_.codec == "h264_nvenc" || cfg_.codec == "nvenc") {
        add_name("h264_nvenc");
    } else {
        // auto: 优先 NVENC（GPU 编码不占 CPU），失败回退 libx264
        add_name("h264_nvenc");
        add_name("libx264");
    }
    if (candidates.empty()) {
        if (auto* c = avcodec_find_encoder(AV_CODEC_ID_H264)) candidates.push_back(c);
    }
    if (candidates.empty()) {
        std::cerr << "[Encoder] No H.264 encoder found" << std::endl;
        return false;
    }

    // 依序尝试每个候选：配置并 open，失败释放上下文继续下一个
    const AVCodec* opened_codec = nullptr;
    for (const AVCodec* codec : candidates) {
        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        if (!ctx) continue;
        ctx->width = width;
        ctx->height = height;
        ctx->time_base = {1, cfg_.fps};
        ctx->framerate = {cfg_.fps, 1};
        ctx->pix_fmt = AV_PIX_FMT_NV12;
        ctx->gop_size = cfg_.gop;
        ctx->bit_rate = static_cast<int64_t>(cfg_.bitrate_kbps) * 1000;
        ctx->max_b_frames = cfg_.max_b_frames;
        // 显式设置色彩空间为 BT.709 limited range，避免播放器误解为 BT.601 导致颜色偏差
        ctx->color_range = AVCOL_RANGE_MPEG;
        ctx->colorspace = AVCOL_SPC_BT709;
        ctx->color_primaries = AVCOL_PRI_BT709;
        ctx->color_trc = AVCOL_TRC_BT709;
        if (codec->id == AV_CODEC_ID_H264) {
            ctx->profile = FF_PROFILE_H264_MAIN;
            ctx->level = 41;
        }

        const bool is_nvenc = (strcmp(codec->name, "h264_nvenc") == 0);
        if (!is_nvenc) {
            // libx264
            av_opt_set(ctx->priv_data, "preset",
                       cfg_.preset.empty() ? "ultrafast" : cfg_.preset.c_str(), 0);
            if (cfg_.low_latency || !cfg_.tune.empty()) {
                av_opt_set(ctx->priv_data, "tune",
                           cfg_.tune.empty() ? "zerolatency" : cfg_.tune.c_str(), 0);
            }
        } else {
            // NVENC
            std::string nv_preset = cfg_.preset;
            if (nv_preset == "ultrafast" || nv_preset.empty()) nv_preset = "p1";
            else if (nv_preset == "superfast") nv_preset = "p2";
            else if (nv_preset == "veryfast") nv_preset = "p3";
            else if (nv_preset == "faster") nv_preset = "p4";
            else if (nv_preset == "fast") nv_preset = "p5";
            else if (nv_preset == "medium") nv_preset = "p5";
            else if (nv_preset == "slow") nv_preset = "p6";
            else if (nv_preset == "slower" || nv_preset == "veryslow") nv_preset = "p7";
            av_opt_set(ctx->priv_data, "preset", nv_preset.c_str(), 0);
            if (cfg_.low_latency) {
                av_opt_set(ctx->priv_data, "tune", "ull", 0);
                av_opt_set(ctx->priv_data, "zerolatency", "1", 0);
            }
        }

        const int open_ret = avcodec_open2(ctx, codec, nullptr);
        if (open_ret >= 0) {
            enc_ctx_ = ctx;
            opened_codec = codec;
            break;
        }
        char errbuf[256] = {};
        av_strerror(open_ret, errbuf, sizeof(errbuf));
        std::cerr << "[Encoder] Failed to open " << codec->name << ": " << errbuf
                  << " (trying next)" << std::endl;
        avcodec_free_context(&ctx);
    }
    if (!enc_ctx_) {
        std::cerr << "[Encoder] All H.264 encoders failed to open" << std::endl;
        return false;
    }

    // 预分配复用 AVFrame 与 AVPacket
    enc_frame_ = av_frame_alloc();
    enc_frame_->width = width;
    enc_frame_->height = height;
    enc_frame_->format = AV_PIX_FMT_NV12;
    enc_frame_->color_range = AVCOL_RANGE_MPEG;
    enc_frame_->colorspace = AVCOL_SPC_BT709;
    enc_frame_->color_primaries = AVCOL_PRI_BT709;
    enc_frame_->color_trc = AVCOL_TRC_BT709;
    if (av_frame_get_buffer(enc_frame_, 32) < 0) {
        std::cerr << "[Encoder] Failed to allocate reusable frame buffer" << std::endl;
        return false;
    }
    enc_pkt_ = av_packet_alloc();

    std::cout << "[Encoder] " << opened_codec->name
              << " [" << width << "x" << height << " @" << cfg_.fps << "fps "
              << cfg_.bitrate_kbps << "kbps gop=" << cfg_.gop << "]" << std::endl;
    return true;
}

bool StreamEncoder::open_output(const std::string& url) {
    // 根据 cfg.format 或 URL 选择输出格式
    std::string fmt = cfg_.format;
    if (fmt.empty() || fmt == "auto") {
        if (url.find("rtsp://") == 0)        fmt = "rtsp";
        else if (url.find("rtmp://") == 0)   fmt = "flv";    // RTMP 容器是 FLV
        else if (url.find("http://") == 0 || url.find("https://") == 0) fmt = "flv";
        else {
            // 按扩展名
            auto pos = url.find_last_of('.');
            if (pos != std::string::npos) {
                std::string ext = url.substr(pos + 1);
                if      (ext == "flv") fmt = "flv";
                else if (ext == "ts")  fmt = "mpegts";
                else if (ext == "mkv") fmt = "matroska";
                else                   fmt = "mp4";
            } else {
                fmt = "mp4";
            }
        }
    }

    avformat_alloc_output_context2(&fmt_ctx_, nullptr, fmt.c_str(), url.c_str());
    if (!fmt_ctx_) {
        std::cerr << "[Encoder] Failed to create output context for: " << url
                  << " (format=" << fmt << ")" << std::endl;
        return false;
    }
    std::cout << "[Encoder] Output format: " << fmt_ctx_->oformat->name << ", url: " << url << std::endl;

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    stream_->time_base = enc_ctx_->time_base;
    avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        int ret = avio_open(&fmt_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "[Encoder] Failed to open output: " << url << " - " << errbuf << std::endl;
            return false;
        }
    }

    int ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[Encoder] Failed to write header: " << errbuf << std::endl;
        return false;
    }
    header_written_ = true;
    opened_ = true;
    std::cout << "[Encoder] Output: " << url << std::endl;
    return true;
}

int64_t StreamEncoder::encode_timed(const modeldeploy::vision::ImageData& image) {
    auto t0 = std::chrono::steady_clock::now();
    bool ok = encode(image);
    auto t1 = std::chrono::steady_clock::now();
    return ok ? std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() : 0;
}

bool StreamEncoder::encode(const modeldeploy::vision::ImageData& image) {
    if (!opened_ || !enc_frame_ || open_permanently_failed_.load()) return false;

    cv::Mat mat;
    image.asMat(&mat);

    if (mat.empty()) {
        std::cerr << "[Encoder] empty input mat" << std::endl;
        return false;
    }

    cv::Mat bgr;
    if (mat.channels() == 4) {
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = mat;
    }

    if (av_frame_make_writable(enc_frame_) < 0) {
        return false;
    }

    const int src_stride = static_cast<int>(bgr.step);
    if (!sws_ctx_ || sws_src_stride_ != src_stride) {
        sws_ctx_ = sws_getCachedContext(sws_ctx_,
            width_, height_, AV_PIX_FMT_BGR24,
            width_, height_, AV_PIX_FMT_NV12,
            SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
        if (sws_ctx_) {
            const int* coef_in  = sws_getCoefficients(SWS_CS_DEFAULT);
            const int* coef_out = sws_getCoefficients(SWS_CS_ITU709);
            sws_setColorspaceDetails(sws_ctx_,
                coef_in,  0,
                coef_out, 0,
                0, 1 << 16, 1 << 16);
        }
        sws_src_stride_ = src_stride;
    }

    const uint8_t* src[1] = { bgr.data };
    int src_stride_arr[1] = { src_stride };
    uint8_t* dst[4] = { enc_frame_->data[0], enc_frame_->data[1], nullptr, nullptr };
    int dst_stride[4] = { enc_frame_->linesize[0], enc_frame_->linesize[1], 0, 0 };
    sws_scale(sws_ctx_, src, src_stride_arr, 0, height_, dst, dst_stride);

    // PTS = 帧序号，同时限速到目标帧率
    // 首帧无条件编码（避免初始化后无数据导致连接超时），后续限速
    if (!accept_frame_rate_limit()) return true;
    enc_frame_->pts = encode_frame_count_;
    ++encode_frame_count_;

    if (avcodec_send_frame(enc_ctx_, enc_frame_) < 0) {
        return false;
    }

    while (avcodec_receive_packet(enc_ctx_, enc_pkt_) == 0) {
        av_packet_rescale_ts(enc_pkt_, enc_ctx_->time_base, stream_->time_base);
        enc_pkt_->stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, enc_pkt_);
        av_packet_unref(enc_pkt_);
    }
    return true;
}

bool StreamEncoder::accept_frame_rate_limit() {
    if (encode_frame_count_ == 0) {
        encode_start_time_ = std::chrono::steady_clock::now();
        return true;
    }
    const double fps = (cfg_.fps > 0) ? cfg_.fps : 25.0;
    const double frame_interval_us = 1000000.0 / fps;
    const double elapsed_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - encode_start_time_).count();
    const int64_t expected_count = static_cast<int64_t>(elapsed_us / frame_interval_us);
    // 已达到或超过预期帧数 → 丢弃本帧（限速到目标帧率）
    return encode_frame_count_ < expected_count;
}

bool StreamEncoder::encode_from_gpu(const uint8_t* gpu_bgr, int width, int height) {
#ifdef WITH_GPU
    if (!opened_ || !enc_frame_ || open_permanently_failed_.load()) return false;
    if (!gpu_bgr || width <= 0 || height <= 0) return false;

    if (av_frame_make_writable(enc_frame_) < 0) {
        return false;
    }

    // 限速（与 encode() 一致）
    if (!accept_frame_rate_limit()) return true;
    enc_frame_->pts = encode_frame_count_;
    ++encode_frame_count_;

    const size_t y_bytes = static_cast<size_t>(height) * width;
    const size_t nv12_bytes = y_bytes * 3 / 2;

    // 复用 device NV12 缓冲
    if (gpu_nv12_capacity_ < nv12_bytes) {
        uint8_t* new_buf = nullptr;
        if (cudaMalloc(&new_buf, nv12_bytes) != cudaSuccess) return false;
        if (gpu_nv12_buf_) cudaFree(gpu_nv12_buf_);
        gpu_nv12_buf_ = new_buf;
        gpu_nv12_capacity_ = nv12_bytes;
    }

    // BGR → NV12（BT.709 limited），gpu_bgr 可为 device 或 host
    if (!modeldeploy::vision::bgr_to_nv12_cuda(gpu_bgr, width, height, gpu_nv12_buf_)) return false;

    // NV12 → enc_frame_：逐平面拷贝，兼容对齐后的 linesize
    cudaError_t c1 = cudaMemcpy2D(enc_frame_->data[0], enc_frame_->linesize[0],
                                  gpu_nv12_buf_, width,
                                  width, height, cudaMemcpyDeviceToHost);
    cudaError_t c2 = cudaMemcpy2D(enc_frame_->data[1], enc_frame_->linesize[1],
                                  gpu_nv12_buf_ + y_bytes, width,
                                  width, height / 2, cudaMemcpyDeviceToHost);
    if (c1 != cudaSuccess || c2 != cudaSuccess) return false;

    if (avcodec_send_frame(enc_ctx_, enc_frame_) < 0) {
        return false;
    }

    while (avcodec_receive_packet(enc_ctx_, enc_pkt_) == 0) {
        av_packet_rescale_ts(enc_pkt_, enc_ctx_->time_base, stream_->time_base);
        enc_pkt_->stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, enc_pkt_);
        av_packet_unref(enc_pkt_);
    }
    return true;
#else
    (void)gpu_bgr; (void)width; (void)height;
    return false;
#endif
}

bool StreamEncoder::encode_from_gpu_nv12(const uint8_t* d_nv12, int width, int height) {
#ifdef WITH_GPU
    if (!opened_ || !enc_frame_ || open_permanently_failed_.load()) return false;
    if (!d_nv12 || width <= 0 || height <= 0) return false;

    if (av_frame_make_writable(enc_frame_) < 0) {
        return false;
    }

    // 限速（与 encode() 一致）
    if (!accept_frame_rate_limit()) return true;
    enc_frame_->pts = encode_frame_count_;
    ++encode_frame_count_;

    const size_t y_bytes = static_cast<size_t>(height) * width;

    // NV12 → enc_frame_：逐平面 D2H 拷贝，兼容对齐后的 linesize。
    // d_nv12 为紧凑连续设备 NV12（step==width），spitch=width。
    // 用每编码器独立非阻塞流异步拷贝：与默认流/解码 D2D/批推理解耦，避免同步
    // 默认流 memcpy 制造 GPU 停滞气泡拖慢整条批推理链路。
    if (!gpu_d2h_stream_) {
        cudaStreamCreateWithFlags(&gpu_d2h_stream_, cudaStreamNonBlocking);
    }
    cudaError_t c1 = cudaMemcpy2DAsync(enc_frame_->data[0], enc_frame_->linesize[0],
                                       d_nv12, width,
                                       width, height, cudaMemcpyDeviceToHost,
                                       gpu_d2h_stream_);
    cudaError_t c2 = cudaMemcpy2DAsync(enc_frame_->data[1], enc_frame_->linesize[1],
                                       d_nv12 + y_bytes, width,
                                       width, height / 2, cudaMemcpyDeviceToHost,
                                       gpu_d2h_stream_);
    if (c1 != cudaSuccess || c2 != cudaSuccess) return false;
    if (cudaStreamSynchronize(gpu_d2h_stream_) != cudaSuccess) return false;

    if (avcodec_send_frame(enc_ctx_, enc_frame_) < 0) {
        return false;
    }

    while (avcodec_receive_packet(enc_ctx_, enc_pkt_) == 0) {
        av_packet_rescale_ts(enc_pkt_, enc_ctx_->time_base, stream_->time_base);
        enc_pkt_->stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, enc_pkt_);
        av_packet_unref(enc_pkt_);
    }
    return true;
#else
    (void)d_nv12; (void)width; (void)height;
    return false;
#endif
}






bool StreamEncoder::start_async() {
    if (async_started_.load()) return true;
    if (!opened_) return false;
    async_running_ = true;
    async_started_ = true;
    encode_thread_ = std::thread(&StreamEncoder::encode_loop, this);
    return true;
}

void StreamEncoder::stop_async() {
    async_running_ = false;
    queue_cv_.notify_all();
    // 排空队列
    drain_queue();
    if (encode_thread_.joinable())
        encode_thread_.join();
    async_started_ = false;
}

void StreamEncoder::drain_queue() {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    std::queue<modeldeploy::vision::ImageData> empty;
    frame_queue_.swap(empty);
}

bool StreamEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    if (!opened_ || !async_running_.load()) return false;
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        if (frame_queue_.size() >= max_queue_size_)
            return false;
        frame_queue_.push(image.clone());
    }
    queue_cv_.notify_one();
    return true;
}

void StreamEncoder::encode_loop() {
    while (async_running_.load()) {
        modeldeploy::vision::ImageData frame;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !frame_queue_.empty() || !async_running_.load(); });
            if (frame_queue_.empty()) continue;
            frame = std::move(frame_queue_.front());
            frame_queue_.pop();
        }
        encode(frame);
    }

    // 退出前处理剩余帧
    while (true) {
        modeldeploy::vision::ImageData frame;
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            if (frame_queue_.empty()) break;
            frame = std::move(frame_queue_.front());
            frame_queue_.pop();
        }
        encode(frame);
    }
}
