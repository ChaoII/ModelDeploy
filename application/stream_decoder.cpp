#include "stream_decoder.hpp"
#include <iostream>
#include <cstring>

static void ff_log_callback(void*, int, const char*, va_list) {}

StreamDecoder::StreamDecoder(const DecoderConfig& cfg) : cfg_(cfg) {
    av_log_set_callback(ff_log_callback);
}

StreamDecoder::~StreamDecoder() {
    stop();
    cleanup();
}

bool StreamDecoder::open(const std::string& url) {
    url_ = url;
    return init_decoder();
}

bool StreamDecoder::start() {
    if (running_.load()) return true;
    if (!fmt_ctx_) return false;
    running_ = true;
    decode_thread_ = std::thread(&StreamDecoder::decode_loop, this);
    return true;
}

void StreamDecoder::stop() {
    running_ = false;
    if (decode_thread_.joinable())
        decode_thread_.join();
}

bool StreamDecoder::init_decoder() {
    cleanup();

    // Open input
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "buffer_size", "65536", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", std::to_string(cfg_.timeout_us).c_str(), 0);

    if (avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts) != 0) {
        av_dict_free(&opts);
        std::cerr << "[Decoder] Failed to open: " << url_ << std::endl;
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

            // Try CUDA hw device
            AVBufferRef* hw_ctx = nullptr;
            if (av_hwdevice_ctx_create(&hw_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) == 0) {
                dec_ctx_->hw_device_ctx = av_buffer_ref(hw_ctx);
                av_buffer_unref(&hw_ctx);
            }

            if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) {
                std::cerr << "[Decoder] Failed to open decoder" << std::endl;
                return false;
            }

            // Get frame rate
            auto* st = fmt_ctx_->streams[i];
            if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)
                fps_ = st->avg_frame_rate.num / st->avg_frame_rate.den;

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
        int ret = av_read_frame(fmt_ctx_, pkt);
        if (ret < 0) {
            // Try reconnect
            if (reconnect_attempts < cfg_.max_reconnects) {
                std::cerr << "[Decoder] Stream error, reconnecting ("
                          << reconnect_attempts + 1 << "/" << cfg_.max_reconnects << ")..." << std::endl;
                reconnect();
                ++reconnect_attempts;
                av_packet_unref(pkt);
                continue;
            }
            std::cerr << "[Decoder] Max reconnects reached, stopping." << std::endl;
            break;
        }
        reconnect_attempts = 0;

        if (pkt->stream_index != video_stream_idx_) {
            av_packet_unref(pkt);
            continue;
        }

        if (avcodec_send_packet(dec_ctx_, pkt) < 0) {
            av_packet_unref(pkt);
            continue;
        }
        av_packet_unref(pkt);

        while (true) {
            ret = avcodec_receive_frame(dec_ctx_, hw_frame);
            if (ret != 0) break;

            // Transfer GPU frame to CPU NV12
            AVFrame* out_frame = hw_frame;
            if (hw_frame->hw_frames_ctx) {
                sw_frame->format = AV_PIX_FMT_NV12;
                sw_frame->width = hw_frame->width;
                sw_frame->height = hw_frame->height;
                av_frame_get_buffer(sw_frame, 0);
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

            if (out_frame == sw_frame)
                av_frame_unref(sw_frame);
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&hw_frame);
    av_frame_free(&sw_frame);
}

void StreamDecoder::reconnect() {
    cleanup();
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.reconnect_delay_ms));

    // Re-open
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "buffer_size", "65536", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "stimeout", std::to_string(cfg_.timeout_us).c_str(), 0);

    if (avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts) == 0) {
        av_dict_free(&opts);
        avformat_find_stream_info(fmt_ctx_, nullptr);
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
    } else {
        av_dict_free(&opts);
    }
}

void StreamDecoder::cleanup() {
    if (dec_ctx_) avcodec_free_context(&dec_ctx_);
    if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
    if (hw_dev_ctx_) av_buffer_unref(&hw_dev_ctx_);
}
