#include "stream_decoder.hpp"
#include <iostream>
#include <cstring>
extern "C" {
#include <libavutil/time.h>
}

static void ff_log_callback(void*, int, const char*, va_list) {}

static bool is_network_url(const std::string& url) {
    return url.find("rtsp://") == 0 || url.find("rtmp://") == 0 ||
           url.find("http://") == 0 || url.find("https://") == 0 ||
           url.find("udp://") == 0 || url.find("tcp://") == 0;
}

StreamDecoder::StreamDecoder(const DecoderConfig& cfg) : cfg_(cfg) {
    av_log_set_callback(ff_log_callback);
}

StreamDecoder::~StreamDecoder() {
    stop();
    cleanup();
}

bool StreamDecoder::open(const std::string& url) {
    std::lock_guard<std::mutex> lock(ctx_mtx_);
    url_ = url;
    return init_decoder();
}

bool StreamDecoder::start() {
    if (running_.load()) return true;
    {
        std::lock_guard<std::mutex> lock(ctx_mtx_);
        if (!fmt_ctx_) return false;
    }
    running_ = true;
    last_read_time_ = av_gettime_relative();
    decode_thread_ = std::thread(&StreamDecoder::decode_loop, this);
    return true;
}

void StreamDecoder::stop() {
    running_ = false;
    if (decode_thread_.joinable())
        decode_thread_.join();
}

int StreamDecoder::interrupt_callback(void* opaque) {
    auto* self = static_cast<StreamDecoder*>(opaque);
    if (!self->running_.load()) return 1; // 请求中断
    // 如果长时间没有读到数据也中断，避免永久阻塞
    int64_t now = av_gettime_relative();
    if (now - self->last_read_time_.load() > self->cfg_.timeout_us) {
        return 1;
    }
    return 0;
}

bool StreamDecoder::init_decoder() {
    cleanup();

    // Open input
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport",
                cfg_.rtsp_transport.empty() ? "tcp" : cfg_.rtsp_transport.c_str(), 0);
    av_dict_set(&opts, "buffer_size", "65536", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", std::to_string(cfg_.timeout_us).c_str(), 0);

    fmt_ctx_ = avformat_alloc_context();
    last_read_time_ = av_gettime_relative();

    int ret = avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts);
    if (ret != 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        av_dict_free(&opts);
        std::cerr << "[Decoder] Failed to open: " << url_ << " - " << errbuf << std::endl;
        return false;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        std::cerr << "[Decoder] Failed to find stream info" << std::endl;
        return false;
    }

    // Find video stream
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            auto* codecpar = fmt_ctx_->streams[i]->codecpar;
            // Try CUVID first, fallback to software
            const AVCodec* dec = avcodec_find_decoder_by_name("h264_cuvid");
            if (!dec) dec = avcodec_find_decoder_by_name("hevc_cuvid");
            if (!dec) dec = avcodec_find_decoder(codecpar->codec_id);
            if (!dec) {
                std::cerr << "[Decoder] No decoder for stream" << std::endl;
                return false;
            }

            dec_ctx_ = avcodec_alloc_context3(dec);
            avcodec_parameters_to_context(dec_ctx_, codecpar);

            // Try CUDA hw device (按 cfg 控制)
            if (cfg_.hw_accel == "cuda") {
                AVBufferRef* hw_ctx = nullptr;
                if (av_hwdevice_ctx_create(&hw_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) == 0) {
                    dec_ctx_->hw_device_ctx = av_buffer_ref(hw_ctx);
                    av_buffer_unref(&hw_ctx);
                }
            }

            if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) {
                std::cerr << "[Decoder] Failed to open decoder" << std::endl;
                return false;
            }

            // Get frame rate
            auto* st = fmt_ctx_->streams[i];
            if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)
                fps_ = st->avg_frame_rate.num / st->avg_frame_rate.den;
            if (fps_ <= 0) fps_ = 25;

            std::cout << "[Decoder] Opened " << url_
                      << " [" << codecpar->width << "x" << codecpar->height
                      << " @" << fps_ << "fps]"
                      << " decoder=" << dec->name
                      << std::endl;
            return true;
        }
    }
    std::cerr << "[Decoder] No video stream found" << std::endl;
    return false;
}

void StreamDecoder::decode_loop() {
    AVPacket* pkt = av_packet_alloc();
    AVFrame* hw_frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();
    int reconnect_attempts = 0;

    while (running_.load()) {
        if (fmt_ctx_) {
            fmt_ctx_->interrupt_callback.callback = &StreamDecoder::interrupt_callback;
            fmt_ctx_->interrupt_callback.opaque = this;
        }
        last_read_time_ = av_gettime_relative();
        int ret = av_read_frame(fmt_ctx_, pkt);
        if (ret < 0) {
            if (!running_.load()) break;
            // EOF on local files is normal termination
            if (ret == AVERROR_EOF && !is_network_url(url_)) {
                std::cout << "[Decoder] End of file reached: " << url_ << std::endl;
                break;
            }
            // For network streams, try reconnect
            if (is_network_url(url_) && reconnect_attempts < cfg_.max_reconnects) {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                std::cerr << "[Decoder] Stream error (" << errbuf << "), reconnecting ("
                          << reconnect_attempts + 1 << "/" << cfg_.max_reconnects << ")..." << std::endl;
                if (reconnect()) {
                    reconnect_attempts = 0;
                } else {
                    ++reconnect_attempts;
                    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.reconnect_delay_ms));
                }
                av_packet_unref(pkt);
                continue;
            }
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "[Decoder] Read error: " << errbuf << ", stopping." << std::endl;
            break;
        }
        reconnect_attempts = 0;

        if (pkt->stream_index != video_stream_idx_) {
            av_packet_unref(pkt);
            continue;
        }

        std::lock_guard<std::mutex> lock(ctx_mtx_);
        if (!dec_ctx_) {
            av_packet_unref(pkt);
            continue;
        }

        if (avcodec_send_packet(dec_ctx_, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }
        av_packet_unref(pkt);

        while (running_.load()) {
            ret = avcodec_receive_frame(dec_ctx_, hw_frame);
            if (ret != 0) break;

            // Transfer GPU frame to CPU NV12
            AVFrame* out_frame = hw_frame;
            if (hw_frame->hw_frames_ctx) {
                av_frame_unref(sw_frame);
                sw_frame->format = AV_PIX_FMT_NV12;
                sw_frame->width = hw_frame->width;
                sw_frame->height = hw_frame->height;
                if (av_frame_get_buffer(sw_frame, 0) < 0) continue;
                if (av_hwframe_transfer_data(sw_frame, hw_frame, 0) == 0)
                    out_frame = sw_frame;
            }

            DecodedFrame df;
            df.y_plane = out_frame->data[0];
            df.uv_plane = out_frame->data[1];
            df.width = out_frame->width;
            df.height = out_frame->height;
            df.y_step = out_frame->linesize[0];
            df.uv_step = out_frame->linesize[1];
            df.pts = out_frame->pts;

            if (callback_ && !callback_(df)) {
                running_ = false;
            }
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&hw_frame);
    av_frame_free(&sw_frame);
}

// ── 同步帧读取（流水线线程直调，不经过 decode_loop） ──

bool StreamDecoder::read_one_frame(DecodedFrame* out) {
    if (!fmt_ctx_ || !out) return false;

    // 延迟分配缓冲
    if (!read_pkt_) read_pkt_ = av_packet_alloc();
    if (!read_hw_frame_) read_hw_frame_ = av_frame_alloc();
    if (!read_sw_frame_) read_sw_frame_ = av_frame_alloc();
    av_frame_unref(read_hw_frame_);
    av_frame_unref(read_sw_frame_);

    while (true) {
        // 先尝试从解码器中取出已缓存的帧
        int ret = avcodec_receive_frame(dec_ctx_, read_hw_frame_);
        if (ret == 0) {
            // 成功拿到一帧
            AVFrame* out_frame = read_hw_frame_;
            if (read_hw_frame_->hw_frames_ctx) {
                // GPU 帧 → 下载到 CPU
                read_sw_frame_->format = AV_PIX_FMT_NV12;
                read_sw_frame_->width = read_hw_frame_->width;
                read_sw_frame_->height = read_hw_frame_->height;
                if (av_frame_get_buffer(read_sw_frame_, 0) < 0) continue;
                if (av_hwframe_transfer_data(read_sw_frame_, read_hw_frame_, 0) == 0)
                    out_frame = read_sw_frame_;
            }
            out->y_plane = out_frame->data[0];
            out->uv_plane = out_frame->data[1];
            out->width = out_frame->width;
            out->height = out_frame->height;
            out->y_step = out_frame->linesize[0];
            out->uv_step = out_frame->linesize[1];
            out->pts = out_frame->pts;
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            // 解码器需要更多数据 → 读下一个 packet
            av_packet_unref(read_pkt_);
            if (fmt_ctx_) {
                fmt_ctx_->interrupt_callback.callback = &StreamDecoder::interrupt_callback;
                fmt_ctx_->interrupt_callback.opaque = this;
            }
            last_read_time_ = av_gettime_relative();
            int r = av_read_frame(fmt_ctx_, read_pkt_);
            if (r < 0) {
                if (r == AVERROR_EOF && !is_network_url(url_)) {
                    return false; // 文件正常结束
                }
                // 网络流错误/EOF → 触发重连
                if (is_network_url(url_) && reconnect_attempts_ < cfg_.max_reconnects) {
                    if (reconnect()) { reconnect_attempts_ = 0; continue; }
                    else { ++reconnect_attempts_; std::this_thread::sleep_for(
                        std::chrono::milliseconds(cfg_.reconnect_delay_ms)); continue; }
                }
                return false;
            }
            reconnect_attempts_ = 0;
            if (read_pkt_->stream_index != video_stream_idx_) { continue; }

            std::lock_guard<std::mutex> lock(ctx_mtx_);
            if (!dec_ctx_) return false;
            if (avcodec_send_packet(dec_ctx_, read_pkt_) < 0) {
                av_packet_unref(read_pkt_);
                continue;
            }
            continue; // 回 loop 头尝试 receive
        }
        // receive 返回其他错误
        return false;
    }
}

bool StreamDecoder::reconnect() {
    std::lock_guard<std::mutex> lock(ctx_mtx_);
    cleanup();

    // Re-open
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "buffer_size", "65536", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", std::to_string(cfg_.timeout_us).c_str(), 0);

    fmt_ctx_ = avformat_alloc_context();
    last_read_time_ = av_gettime_relative();

    if (avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts) == 0) {
        av_dict_free(&opts);
        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            cleanup();
            return false;
        }
        // Re-init decoder
        for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
            if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_idx_ = i;
                const AVCodec* dec = avcodec_find_decoder(fmt_ctx_->streams[i]->codecpar->codec_id);
                if (dec) {
                    dec_ctx_ = avcodec_alloc_context3(dec);
                    avcodec_parameters_to_context(dec_ctx_, fmt_ctx_->streams[i]->codecpar);
                    avcodec_open2(dec_ctx_, dec, nullptr);
                }
                break;
            }
        }
        return dec_ctx_ != nullptr;
    }
    av_dict_free(&opts);
    return false;
}

void StreamDecoder::cleanup() {
    if (dec_ctx_) avcodec_free_context(&dec_ctx_);
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    if (hw_dev_ctx_) av_buffer_unref(&hw_dev_ctx_);
    video_stream_idx_ = -1;
}
