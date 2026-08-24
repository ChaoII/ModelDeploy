#include "csrc/video/backend/ffmpeg_decoder.h"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

namespace modeldeploy::video {

using modeldeploy::Device;

FfmpegDecoder::FfmpegDecoder(const VideoDecoderConfig& cfg)
    : cfg_(cfg), state_(State::Idle) {}

FfmpegDecoder::~FfmpegDecoder() { cleanup(); }

bool FfmpegDecoder::runtime_available() const { return true; }

bool FfmpegDecoder::open(const std::string& url, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    cleanup();
    AVDictionary* opts = nullptr;
    std::string timeout_str = std::to_string(cfg_.timeout_us);
    av_dict_set(&opts, "stimeout", timeout_str.c_str(), 0);
    fmt_ = avformat_alloc_context();
    state_ = State::Opening;
    if (avformat_open_input(&fmt_, url.c_str(), nullptr, &opts) < 0) {
        av_dict_free(&opts);
        set_err(err, "open-input-fail");
        cleanup();
        state_ = State::Error;
        return false;
    }
    av_dict_free(&opts);
    if (avformat_find_stream_info(fmt_, nullptr) < 0) {
        set_err(err, "find-stream-info-fail");
        cleanup();
        state_ = State::Error;
        return false;
    }
    for (unsigned i = 0; i < fmt_->nb_streams; ++i) {
        auto* cp = fmt_->streams[i]->codecpar;
        if (cp->codec_type != AVMEDIA_TYPE_VIDEO) continue;
        const AVCodec* dec = avcodec_find_decoder(cp->codec_id);   // 仅软解
        if (!dec) {
            set_err(err, "no-soft-decoder");
            cleanup();
            state_ = State::Error;
            return false;
        }
        vstream_ = static_cast<int>(i);
        ctx_ = avcodec_alloc_context3(dec);
        avcodec_parameters_to_context(ctx_, cp);
        if (avcodec_open2(ctx_, dec, nullptr) < 0) {
            set_err(err, "decoder-open-fail");
            cleanup();
            state_ = State::Error;
            return false;
        }
        w_ = cp->width;
        h_ = cp->height;
        auto* st = fmt_->streams[i];
        if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)
            fps_ = static_cast<double>(st->avg_frame_rate.num) / st->avg_frame_rate.den;
        if (fps_ <= 0.0) fps_ = 25.0;
        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        opened_ = true;
        state_ = State::Running;
        return true;
    }
    set_err(err, "no-video-stream");
    cleanup();
    state_ = State::Error;
    return false;
}

bool FfmpegDecoder::read_one_frame(VideoFrame* out, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_ || !fmt_ || !ctx_ || !out) {
        set_err(err, "not-opened");
        return false;
    }
    while (true) {
        int ret = avcodec_receive_frame(ctx_, frame_);
        if (ret == 0) {
            // 解码器输出可能是 NV12（硬解/部分软解）或 yuv420p 等。非 NV12 一律经
            // swscale 转成 NV12，避免用 data[0]/data[1] 直接构造平面时静默丢掉 V 平面。
            AVFrame* plane_src = nullptr;
            if (frame_->format == AV_PIX_FMT_NV12) {
                plane_src = frame_;
            } else {
                if (!convert_to_nv12()) {
                    set_err(err, "nv12-convert-fail");
                    return false;
                }
                plane_src = sws_frame_;
            }
            if (!plane_src->data[0] || !plane_src->data[1]) {
                set_err(err, "no-nv12");
                return false;
            }
            // 把（可能的）swscale 结果 ref 到自有 owner，保证 ImageData 生命周期内缓冲有效
            std::shared_ptr<AVFrame> owned(av_frame_alloc(),
                                           [](AVFrame* f) { av_frame_free(&f); });
            if (av_frame_ref(owned.get(), plane_src) < 0) {
                set_err(err, "ref-fail");
                return false;
            }
            IPlaneView v{owned->data[0], owned->linesize[0],
                         owned->data[1], owned->linesize[1],
                         owned->width, owned->height, Device::CPU, owned};
            out->image = make_image_from_planes_view(v);
            auto* st = fmt_->streams[vstream_];
            out->pts_ms = (frame_->pts == AV_NOPTS_VALUE)
                              ? 0
                              : static_cast<uint64_t>(
                                    av_rescale_q(frame_->pts, st->time_base, AVRational{1, 1000}));
            stats_.frames_out++;
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            av_packet_unref(pkt_);
            int r = av_read_frame(fmt_, pkt_);
            if (r < 0) {  // EOF/错误（本地文件正常结束）
                set_err(err, "eof");
                return false;
            }
            if (pkt_->stream_index != vstream_) {
                av_packet_unref(pkt_);
                continue;
            }
            if (avcodec_send_packet(ctx_, pkt_) < 0) {
                av_packet_unref(pkt_);
                continue;
            }
            stats_.frames_in++;
            continue;
        }
        set_err(err, "receive-frame-error");
        return false;
    }
}

bool FfmpegDecoder::convert_to_nv12() {
    if (!sws_frame_) sws_frame_ = av_frame_alloc();
    if (!sws_frame_) return false;
    if (!sws_frame_->buf[0]) {
        // 首次分配 NV12 双平面 buffer（Y 与 UV 同块，data[1] 由 AVFrame 推导）
        sws_frame_->format = AV_PIX_FMT_NV12;
        sws_frame_->width = w_;
        sws_frame_->height = h_;
        if (av_frame_get_buffer(sws_frame_, 128) < 0) return false;
    }
    if (!sws_ || sws_fmt_ != frame_->format) {
        sws_freeContext(sws_);
        sws_ = sws_getContext(
            frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format),
            w_, h_, AV_PIX_FMT_NV12,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_) return false;
        sws_fmt_ = frame_->format;
    }
    sws_scale(sws_,
              frame_->data, frame_->linesize, 0, frame_->height,
              sws_frame_->data, sws_frame_->linesize);
    sws_frame_->width = w_;
    sws_frame_->height = h_;
    return true;
}

void FfmpegDecoder::set_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    // Phase1 后端内最小实现：记录回调（异步推送留给 Phase2 线程化运行时）
    (void)cb;
}

bool FfmpegDecoder::start(std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_) {
        set_err(err, "not-opened");
        return false;
    }
    state_ = State::Running;
    return true;
}

void FfmpegDecoder::stop() {
    std::lock_guard<std::mutex> lk(mtx_);
    state_ = State::Idle;
}

void FfmpegDecoder::set_device_only(bool v) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_.device_only = v;  // 设备直通留 Phase2
}

int FfmpegDecoder::fps() const { return static_cast<int>(fps_); }

int FfmpegDecoder::width() const { return w_; }

int FfmpegDecoder::height() const { return h_; }

VideoStats& FfmpegDecoder::stats() { return stats_; }

std::string FfmpegDecoder::last_error() const { return err_; }

void FfmpegDecoder::close() {
    std::lock_guard<std::mutex> lk(mtx_);
    cleanup();
    state_ = State::Closed;
}

void FfmpegDecoder::cleanup() {
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (sws_frame_) av_frame_free(&sws_frame_);
    sws_freeContext(sws_);
    sws_ = nullptr;
    sws_fmt_ = -1;
    if (ctx_) avcodec_free_context(&ctx_);
    if (fmt_) {
        avformat_close_input(&fmt_);
        fmt_ = nullptr;
    }
    vstream_ = -1;
    w_ = 0;
    h_ = 0;
    opened_ = false;
}

void FfmpegDecoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

} // namespace modeldeploy::video
