#include "stream_encoder.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <cstring>

StreamEncoder::StreamEncoder(int fps) : fps_(fps) {
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
        avcodec_send_frame(enc_ctx_, nullptr);
        AVPacket pkt;
        av_init_packet(&pkt);
        while (avcodec_receive_packet(enc_ctx_, &pkt) == 0)
            av_packet_unref(&pkt);
    }
    if (sws_ctx_) sws_freeContext(sws_ctx_);
    if (fmt_ctx_) {
        av_write_trailer(fmt_ctx_);
        if (fmt_ctx_->pb)
            avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
    }
    if (enc_ctx_) avcodec_free_context(&enc_ctx_);
    sws_ctx_ = nullptr;
    fmt_ctx_ = nullptr;
    enc_ctx_ = nullptr;
    opened_ = false;
}

bool StreamEncoder::init_encoder(int width, int height) {
    auto* codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "[Encoder] No H.264 encoder found" << std::endl;
        return false;
    }

    enc_ctx_ = avcodec_alloc_context3(codec);
    enc_ctx_->width = width;
    enc_ctx_->height = height;
    enc_ctx_->time_base = {1, fps_};
    enc_ctx_->framerate = {fps_, 1};
    enc_ctx_->pix_fmt = AV_PIX_FMT_NV12;
    enc_ctx_->gop_size = fps_ * 2;
    enc_ctx_->bit_rate = 8'000'000;
    enc_ctx_->max_b_frames = 2;
    if (codec->id == AV_CODEC_ID_H264) {
        enc_ctx_->profile = FF_PROFILE_H264_MAIN;
        enc_ctx_->level = 41;
    }

    // Optimize for low-latency encoding
    if (codec->id == AV_CODEC_ID_H264) {
        // Software x264 - ultrafast preset + zerolatency
        av_opt_set(enc_ctx_->priv_data, "preset", "ultrafast", 0);
        av_opt_set(enc_ctx_->priv_data, "tune", "zerolatency", 0);
        enc_ctx_->bit_rate = 6'000'000;
        std::cout << "[Encoder] x264 ultrafast+zerolatency" << std::endl;
    }

    if (avcodec_open2(enc_ctx_, codec, nullptr) < 0) {
        std::cerr << "[Encoder] Failed to open encoder" << std::endl;
        return false;
    }

    std::cout << "[Encoder] " << codec->name
              << " [" << width << "x" << height << " @" << fps_ << "fps]" << std::endl;
    return true;
}

bool StreamEncoder::open_output(const std::string& url) {
    avformat_alloc_output_context2(&fmt_ctx_, nullptr, "rtsp", url.c_str());
    if (!fmt_ctx_)
        avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", url.c_str());
    if (!fmt_ctx_) {
        std::cerr << "[Encoder] Failed to create output context" << std::endl;
        return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    stream_->time_base = enc_ctx_->time_base;
    avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE) < 0) {
            std::cerr << "[Encoder] Failed to open output: " << url << std::endl;
            return false;
        }
    }

    avformat_write_header(fmt_ctx_, nullptr);
    opened_ = true;
    std::cout << "[Encoder] Output: " << url << std::endl;
    return true;
}

bool StreamEncoder::encode(const modeldeploy::vision::ImageData& image) {
    if (!opened_) return false;
    cv::Mat mat;
    image.to_mat(mat, false);

    auto* frame = av_frame_alloc();
    frame->width = width_;
    frame->height = height_;
    frame->format = AV_PIX_FMT_NV12;
    av_frame_get_buffer(frame, 32);

    sws_ctx_ = sws_getCachedContext(sws_ctx_,
        width_, height_, AV_PIX_FMT_BGR24,
        width_, height_, AV_PIX_FMT_NV12,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    const uint8_t* src[1] = { mat.data };
    int src_stride[1] = { static_cast<int>(mat.step) };
    uint8_t* dst[4] = { frame->data[0], frame->data[1], nullptr, nullptr };
    int dst_stride[4] = { frame->linesize[0], frame->linesize[1], 0, 0 };
    sws_scale(sws_ctx_, src, src_stride, 0, height_, dst, dst_stride);

    bool ok = encode_frame(frame);
    av_frame_free(&frame);
    return ok;
}

bool StreamEncoder::encode_frame(AVFrame* frame) {
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;

    frame->pts = frame_pts_++;

    if (avcodec_send_frame(enc_ctx_, frame) < 0)
        return false;

    while (avcodec_receive_packet(enc_ctx_, &pkt) == 0) {
        av_packet_rescale_ts(&pkt, enc_ctx_->time_base, stream_->time_base);
        pkt.stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, &pkt);
        av_packet_unref(&pkt);
    }
    return true;
}

bool StreamEncoder::start_async() {
    if (async_running_) return true;
    if (!opened_) return false;
    async_running_ = true;
    encode_thread_ = std::thread(&StreamEncoder::encode_loop, this);
    return true;
}

void StreamEncoder::stop_async() {
    async_running_ = false;
    queue_cv_.notify_all();
    if (encode_thread_.joinable())
        encode_thread_.join();
}

bool StreamEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    if (!opened_ || !async_running_) return false;
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
    while (async_running_) {
        modeldeploy::vision::ImageData frame;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                [this]() { return !frame_queue_.empty() || !async_running_; });
            if (frame_queue_.empty()) continue;
            frame = std::move(frame_queue_.front());
            frame_queue_.pop();
        }
        encode(frame);
    }
}
