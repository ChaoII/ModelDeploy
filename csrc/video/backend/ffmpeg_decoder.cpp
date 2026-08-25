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
    force_soft_ = false;  // 外部全新 open：复位强制软解标志（允许重新走硬件）
    return open_locked(url, err, /*reset_fallback=*/true);
}

// 打开核心：假设调用方已持有 mtx_。force_soft_ 为 true 时强制走软解路径（运行期回退重开）。
bool FfmpegDecoder::open_locked(const std::string& url, std::string* err, bool reset_fallback) {
    cleanup();
    if (reset_fallback) {
        delivered_frames_ = 0;
        hw_fallback_done_ = false;
    }
    url_ = url;
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
        // 硬解选择：仅已显式/自动请求 CUDA 且未强制软解时尝试 CUVID；否则（含降级）回软解。
        bool used_hw = false;
        bool want_hw = !force_soft_ &&
                       (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda);
        if (want_hw && cfg_.device_only) {
            // 设备直通：CUDA 硬解 → AV_PIX_FMT_CUDA 设备帧，绝不出主机。
            const std::string hw_name = hw_decoder_name(cp->codec_id);
            const AVCodec* hwc =
                hw_name.empty() ? nullptr : avcodec_find_decoder_by_name(hw_name.c_str());
            if (hwc && setup_cuda_device_decoder(cp, hwc)) {
                used_hw = true;
                device_only_active_ = true;
            } else {
                // 设备直通不可行（无 CUDA 硬件/解码器/设备帧输出）报设备失败，供上层 SKIP 判定。
                err_ = "cuda-device-unavailable";
            }
        } else if (want_hw && !cfg_.device_only) {
            std::string hw_name = hw_decoder_name(cp->codec_id);
            const AVCodec* hwc = nullptr;
            if (!hw_name.empty()) hwc = avcodec_find_decoder_by_name(hw_name.c_str());
            if (hwc) {
                ctx_ = avcodec_alloc_context3(hwc);
                avcodec_parameters_to_context(ctx_, cp);
                if (avcodec_open2(ctx_, hwc, nullptr) == 0) {
                    used_hw = true;  // 硬解成功：h264_cuvid 直接输出 CPU NV12，走 IPlaneView 分支
                } else {
                    avcodec_free_context(&ctx_);
                    ctx_ = nullptr;
                    err_ = "hw-decoder-open-fail-fallback-soft";  // 降级：不计 error_count
                }
            } else {
                err_ = "hw-decoder-not-found-fallback-soft";  // 无对应 CUVID 名/解码器：降级软解
            }
        }
        // 设备直通模式不提供软解回退：解码器打开为设备帧是硬性要求，失败即整体失败。
        if (cfg_.device_only) {
            if (!used_hw) {
                set_err(err, "cuda-device-unavailable");
                cleanup();
                state_ = State::Error;
                return false;
            }
        } else if (!used_hw) {
            // 软解（默认路径或硬解降级）。cuvid 装不上/不存在不挂会话，回退软解。
            const AVCodec* dec = avcodec_find_decoder(cp->codec_id);
            if (!dec) {
                set_err(err, "no-soft-decoder");
                cleanup();
                state_ = State::Error;
                return false;
            }
            ctx_ = avcodec_alloc_context3(dec);
            avcodec_parameters_to_context(ctx_, cp);
            if (avcodec_open2(ctx_, dec, nullptr) < 0) {
                set_err(err, "decoder-open-fail");
                cleanup();
                state_ = State::Error;
                return false;
            }
        }
        used_hw_ = used_hw;
        vstream_ = static_cast<int>(i);
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
    return read_one_frame_locked(out, err);
}

// 读一帧核心：假设调用方已持有 mtx_。读到 EOF 且本届走硬解但尚未交付任何帧时，
// （至多一次）自动重开为软解并重入本函数继续读取，透明救回"硬解 0 帧"场景。
bool FfmpegDecoder::read_one_frame_locked(VideoFrame* out, std::string* err) {
    if (!opened_ || !fmt_ || !ctx_ || !out) {
        set_err(err, "not-opened");
        return false;
    }
    while (true) {
        int ret = avcodec_receive_frame(ctx_, frame_);
        if (ret == 0) {
            if (device_only_active_) {
                // 设备直通：解码帧必须是 AV_PIX_FMT_CUDA（CUDA 设备内存，NV12 双平面）。
                // data[0]=Y / data[1]=UV 为设备可寻址指针，linesize[] 为各平面步长，不回主机。
                if (frame_->format != AV_PIX_FMT_CUDA || !frame_->data[0] || !frame_->data[1]) {
                    set_err(err, "device-frame-unavailable");
                    return false;
                }
                // 把 hw frame ref 到自有 owner，保证 ImageData 生命周期内设备缓冲有效
                std::shared_ptr<AVFrame> owned(av_frame_alloc(),
                                               [](AVFrame* f) { av_frame_free(&f); });
                if (av_frame_ref(owned.get(), frame_) < 0) {
                    set_err(err, "ref-fail");
                    return false;
                }
                IPlaneView v{owned->data[0], owned->linesize[0],
                             owned->data[1], owned->linesize[1],
                             (int)owned->width, (int)owned->height, Device::GPU, owned};
                out->image = make_image_from_planes_view(v);
                auto* st = fmt_->streams[vstream_];
                out->pts_ms = (frame_->pts == AV_NOPTS_VALUE)
                                  ? 0
                                  : static_cast<uint64_t>(
                                        av_rescale_q(frame_->pts, st->time_base, AVRational{1, 1000}));
                stats_.frames_out++;
                delivered_frames_++;
                return true;
            }
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
            delivered_frames_++;
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            av_packet_unref(pkt_);
            int r = av_read_frame(fmt_, pkt_);
            if (r < 0) {  // EOF/错误（本地文件正常结束）
                // 运行期 0 帧回退：本届确实走了硬解（used_hw_）却尚未交付任何一帧就 EOF
                // （如 cuvid 运行时对特殊分辨率 CUDA_ERROR_NOT_SUPPORTED），且尚未回退过 →
                // 释放当前上下文，以强制软解重开同一 URL 并继续读取（降级，不计 error_count）。
                // 设备直通模式下不做软解回退（device_only 要求输出设备帧）。
                if (used_hw_ && !device_only_active_ && delivered_frames_ == 0 && !hw_fallback_done_) {
                    hw_fallback_done_ = true;      // 至多一次，防死循环
                    cleanup();                     // 已持锁：cleanup()/open_locked() 均不重复加锁
                    force_soft_ = true;            // 本次重开强制走软解
                    std::string reopen_err;
                    if (!open_locked(url_, &reopen_err, /*reset_fallback=*/false)) {
                        err_ = "hw-fallback-reopen-fail";
                        if (err) *err = err_;
                        return false;
                    }
                    // 重开成功（软解）：重入继续读取，把软解结果交付给调用方
                    return read_one_frame_locked(out, err);
                }
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

std::string FfmpegDecoder::hw_decoder_name(int codec_id) const {
    switch (codec_id) {
        case AV_CODEC_ID_H264: return "h264_cuvid";
        case AV_CODEC_ID_HEVC: return "hevc_cuvid";
        case AV_CODEC_ID_AV1: return "av1_cuvid";
        default: return "";  // 其它编解码器无 CUVID 硬解名
    }
}

bool FfmpegDecoder::setup_cuda_device_decoder(AVCodecParameters* cp, const AVCodec* hwc) {
    AVBufferRef* hw = nullptr;
    if (av_hwdevice_ctx_create(&hw, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) != 0) return false;
    ctx_ = avcodec_alloc_context3(hwc);
    if (!ctx_) {
        av_buffer_unref(&hw);
        return false;
    }
    avcodec_parameters_to_context(ctx_, cp);
    // 挂上 CUDA 硬件设备上下文：h264_cuvid 等解码器据此输出 AV_PIX_FMT_CUDA 设备帧
    ctx_->hw_device_ctx = av_buffer_ref(hw);
    av_buffer_unref(&hw);  // ctx_ 持有一份引用
    if (avcodec_open2(ctx_, hwc, nullptr) != 0) {
        avcodec_free_context(&ctx_);
        ctx_ = nullptr;
        return false;
    }
    return true;
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
    used_hw_ = false;
    device_only_active_ = false;
}

void FfmpegDecoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

} // namespace modeldeploy::video
