#include "csrc/video/video_decoder.h"
#include <memory>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}

namespace modeldeploy::video {

using modeldeploy::vision::ImageData;

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open(const std::string& url) {
    cleanup();
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "stimeout", "5000000", 0);
    fmt_ctx_ = avformat_alloc_context();
    if (avformat_open_input(&fmt_ctx_, url.c_str(), nullptr, &opts) < 0) {
        av_dict_free(&opts);
        cleanup();
        return false;
    }
    av_dict_free(&opts);
    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        cleanup();
        return false;
    }
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        auto* cp = fmt_ctx_->streams[i]->codecpar;
        if (cp->codec_type != AVMEDIA_TYPE_VIDEO) continue;
        const AVCodec* dec = avcodec_find_decoder(cp->codec_id);   // 仅软解
        if (!dec) { cleanup(); return false; }
        video_stream_idx_ = static_cast<int>(i);
        dec_ctx_ = avcodec_alloc_context3(dec);
        avcodec_parameters_to_context(dec_ctx_, cp);
        if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) { cleanup(); return false; }
        width_ = cp->width;
        height_ = cp->height;
        auto* st = fmt_ctx_->streams[i];
        if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)
            fps_ = static_cast<double>(st->avg_frame_rate.num) / st->avg_frame_rate.den;
        if (fps_ <= 0.0) fps_ = 25.0;
        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        return true;
    }
    cleanup();
    return false;
}

bool VideoDecoder::next(modeldeploy::vision::ImageData* out, uint64_t* pts_ms) {
    if (!fmt_ctx_ || !dec_ctx_ || !out) return false;
    while (true) {
        int ret = avcodec_receive_frame(dec_ctx_, frame_);
        if (ret == 0) {
            // 解码器输出可能是 NV12（h264 硬解/部分软解）或 yuv420p 等。非 NV12 一律经
            // swscale 转成 NV12，避免用 data[0]/data[1] 直接构造平面时静默丢掉 V 平面。
            AVFrame* plane_src = nullptr;
            if (frame_->format == AV_PIX_FMT_NV12) {
                plane_src = frame_;
            } else {
                if (!convert_to_nv12()) return false;
                plane_src = sws_frame_;
            }
            if (!plane_src->data[0] || !plane_src->data[1]) return false;  // NV12 双平面
            // 把（可能的）swscale 结果 ref 到自有 owner，保证 ImageData 生命周期内缓冲有效
            std::shared_ptr<AVFrame> owned(av_frame_alloc(),
                                           [](AVFrame* f) { av_frame_free(&f); });
            if (av_frame_ref(owned.get(), plane_src) < 0) return false;
            ImageData::Plane pl[2] = {
                {owned->data[0], owned->linesize[0]},
                {owned->data[1], owned->linesize[1]},
            };
            *out = ImageData::from_planes(pl, 2, MdImageType::NV12,
                                          owned->width, owned->height,
                                          modeldeploy::Device::CPU, owned);
            if (pts_ms) {
                auto* st = fmt_ctx_->streams[video_stream_idx_];
                *pts_ms = (frame_->pts == AV_NOPTS_VALUE)
                              ? 0
                              : static_cast<uint64_t>(
                                    av_rescale_q(frame_->pts, st->time_base, AVRational{1, 1000}));
            }
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            av_packet_unref(pkt_);
            int r = av_read_frame(fmt_ctx_, pkt_);
            if (r < 0) return false;  // EOF/错误（本地文件正常结束）
            if (pkt_->stream_index != video_stream_idx_) { av_packet_unref(pkt_); continue; }
            if (avcodec_send_packet(dec_ctx_, pkt_) < 0) { av_packet_unref(pkt_); continue; }
            continue;
        }
        return false;
    }
}

bool VideoDecoder::convert_to_nv12() {
    if (!sws_frame_) sws_frame_ = av_frame_alloc();
    if (!sws_frame_) return false;
    if (!sws_frame_->buf[0]) {
        // 首次分配 NV12 双平面 buffer（Y 与 UV 同块，数据[1] 由 AVFrame 推导）
        sws_frame_->format = AV_PIX_FMT_NV12;
        sws_frame_->width = width_;
        sws_frame_->height = height_;
        if (av_frame_get_buffer(sws_frame_, 128) < 0) return false;
    }
    if (!sws_ctx_ || sws_src_fmt_ != frame_->format) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = sws_getContext(
            frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format),
            width_, height_, AV_PIX_FMT_NV12,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx_) return false;
        sws_src_fmt_ = frame_->format;
    }
    sws_scale(sws_ctx_,
              frame_->data, frame_->linesize, 0, frame_->height,
              sws_frame_->data, sws_frame_->linesize);
    sws_frame_->width = width_;
    sws_frame_->height = height_;
    return true;
}

void VideoDecoder::close() { cleanup(); }

void VideoDecoder::cleanup() {
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (sws_frame_) av_frame_free(&sws_frame_);
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
    sws_src_fmt_ = -1;
    if (dec_ctx_) avcodec_free_context(&dec_ctx_);
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_idx_ = -1;
    width_ = 0;
    height_ = 0;
}

} // namespace modeldeploy::video
