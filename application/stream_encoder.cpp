#include "stream_encoder.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>

StreamEncoder::StreamEncoder(const EncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.fps <= 0) cfg_.fps = 25;
    if (cfg_.bitrate_kbps <= 0) cfg_.bitrate_kbps = 4000;
    if (cfg_.gop <= 0) cfg_.gop = cfg_.fps * 2;
}

StreamEncoder::~StreamEncoder() {
    stop_async();
    close();
}

bool StreamEncoder::open(const std::string& output_url, int width, int height) {
    close();
    width_ = width;
    height_ = height;
    if (!init_encoder(width, height)) return false;
    return open_output(output_url);
}

void StreamEncoder::close() {
    if (enc_ctx_) {
        // 发送 nullptr 触发编码器刷新
        avcodec_send_frame(enc_ctx_, nullptr);
        AVPacket* pkt = av_packet_alloc();
        while (avcodec_receive_packet(enc_ctx_, pkt) == 0) {
            av_packet_rescale_ts(pkt, enc_ctx_->time_base, stream_->time_base);
            pkt->stream_index = stream_->index;
            av_interleaved_write_frame(fmt_ctx_, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }
    if (sws_ctx_) sws_freeContext(sws_ctx_);
    if (fmt_ctx_) {
        av_write_trailer(fmt_ctx_);
        if (fmt_ctx_->pb && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
    }
    if (enc_ctx_) avcodec_free_context(&enc_ctx_);
    sws_ctx_ = nullptr;
    fmt_ctx_ = nullptr;
    enc_ctx_ = nullptr;
    stream_ = nullptr;
    opened_ = false;
    frame_pts_ = 0;
}

bool StreamEncoder::init_encoder(int width, int height) {
    // 编码器选择：cfg.codec=auto 时优先 libx264 → h264_nvenc → 默认 H264
    const AVCodec* codec = nullptr;
    if (cfg_.codec == "libx264" || cfg_.codec == "x264") {
        codec = avcodec_find_encoder_by_name("libx264");
    } else if (cfg_.codec == "h264_nvenc" || cfg_.codec == "nvenc") {
        codec = avcodec_find_encoder_by_name("h264_nvenc");
    } else {
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) codec = avcodec_find_encoder_by_name("h264_nvenc");
    }
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "[Encoder] No H.264 encoder found" << std::endl;
        return false;
    }

    enc_ctx_ = avcodec_alloc_context3(codec);
    enc_ctx_->width = width;
    enc_ctx_->height = height;
    enc_ctx_->time_base = {1, cfg_.fps};
    enc_ctx_->framerate = {cfg_.fps, 1};
    enc_ctx_->pix_fmt = AV_PIX_FMT_NV12;
    enc_ctx_->gop_size = cfg_.gop;
    enc_ctx_->bit_rate = static_cast<int64_t>(cfg_.bitrate_kbps) * 1000;
    enc_ctx_->max_b_frames = cfg_.max_b_frames;
    if (codec->id == AV_CODEC_ID_H264) {
        enc_ctx_->profile = FF_PROFILE_H264_MAIN;
        enc_ctx_->level = 41;
    }

    // 编码器特定参数
    bool is_nvenc = (strcmp(codec->name, "h264_nvenc") == 0);
    if (!is_nvenc) {
        // libx264
        av_opt_set(enc_ctx_->priv_data, "preset",
                   cfg_.preset.empty() ? "ultrafast" : cfg_.preset.c_str(), 0);
        if (cfg_.low_latency || !cfg_.tune.empty()) {
            av_opt_set(enc_ctx_->priv_data, "tune",
                       cfg_.tune.empty() ? "zerolatency" : cfg_.tune.c_str(), 0);
        }
        std::cout << "[Encoder] libx264 preset=" << cfg_.preset
                  << " tune=" << cfg_.tune << std::endl;
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
        av_opt_set(enc_ctx_->priv_data, "preset", nv_preset.c_str(), 0);
        if (cfg_.low_latency) {
            av_opt_set(enc_ctx_->priv_data, "tune", "ull", 0);
            av_opt_set(enc_ctx_->priv_data, "zerolatency", "1", 0);
        }
        std::cout << "[Encoder] NVENC preset=" << nv_preset
                  << " low_latency=" << cfg_.low_latency << std::endl;
    }

    int ret = avcodec_open2(enc_ctx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[Encoder] Failed to open encoder: " << errbuf << std::endl;
        return false;
    }

    std::cout << "[Encoder] " << codec->name
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
    opened_ = true;
    std::cout << "[Encoder] Output: " << url << std::endl;
    return true;
}

bool StreamEncoder::encode(const modeldeploy::vision::ImageData& image) {
    if (!opened_) return false;

    cv::Mat mat;
    image.to_mat(mat, true); // 强制深拷贝到 CPU
    if (mat.empty()) {
        std::cerr << "[Encoder] empty input mat" << std::endl;
        return false;
    }

    // 确保 BGR 三通道
    cv::Mat bgr;
    if (mat.channels() == 4) {
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = mat;
    }

    auto* frame = av_frame_alloc();
    frame->width = width_;
    frame->height = height_;
    frame->format = AV_PIX_FMT_NV12;
    if (av_frame_get_buffer(frame, 32) < 0) {
        av_frame_free(&frame);
        return false;
    }

    sws_ctx_ = sws_getCachedContext(sws_ctx_,
        width_, height_, AV_PIX_FMT_BGR24,
        width_, height_, AV_PIX_FMT_NV12,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    const uint8_t* src[1] = { bgr.data };
    int src_stride[1] = { static_cast<int>(bgr.step) };
    uint8_t* dst[4] = { frame->data[0], frame->data[1], nullptr, nullptr };
    int dst_stride[4] = { frame->linesize[0], frame->linesize[1], 0, 0 };
    sws_scale(sws_ctx_, src, src_stride, 0, height_, dst, dst_stride);

    bool ok = encode_frame(frame);
    av_frame_free(&frame);
    return ok;
}

bool StreamEncoder::encode_frame(AVFrame* frame) {
    AVPacket* pkt = av_packet_alloc();

    frame->pts = frame_pts_++;

    if (avcodec_send_frame(enc_ctx_, frame) < 0) {
        av_packet_free(&pkt);
        return false;
    }

    while (avcodec_receive_packet(enc_ctx_, pkt) == 0) {
        av_packet_rescale_ts(pkt, enc_ctx_->time_base, stream_->time_base);
        pkt->stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, pkt);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    return true;
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
