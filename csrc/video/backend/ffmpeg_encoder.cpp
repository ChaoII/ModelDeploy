#include "csrc/video/backend/ffmpeg_encoder.h"

extern "C" {
#include <libavutil/opt.h>
}

#include <chrono>
#include <vector>

namespace modeldeploy::video {

FfmpegEncoder::FfmpegEncoder(const VideoEncoderConfig& cfg) : cfg_(cfg) {}

FfmpegEncoder::~FfmpegEncoder() { cleanup(); }

bool FfmpegEncoder::runtime_available() const {
    return avcodec_find_encoder_by_name("libx264") != nullptr;
}

bool FfmpegEncoder::open(const std::string& url, int w, int h, int src_fps,
                         const VideoEncoderConfig& c, std::string* err) {
    cleanup();
    cfg_ = c;
    w_ = w;
    h_ = h;
    // fps==0 视为自动：优先取配置，其次取源帧率，最后兜底 25
    if (cfg_.fps <= 0) cfg_.fps = (src_fps > 0) ? src_fps : 25;
    if (cfg_.gop <= 0) cfg_.gop = cfg_.fps * 2;
    if (w <= 0 || h <= 0) {
        set_err(err, "invalid-dimension");
        return false;
    }
    // 仅 libx264 可用性兜底检查（显式 h264_nvenc 在 init_encoder 内自行判定，不在此拦截）
    bool want_nvenc = (cfg_.codec == "h264_nvenc");
    if (!want_nvenc && !runtime_available()) {
        set_err(err, "no-libx264");
        return false;
    }
    if (!init_encoder(w, h, cfg_.fps, err)) {
        cleanup();  // init_encoder 部分分配的资源在此统一释放，避免滞留
        return false;
    }
    if (!open_output(url)) {
        set_err(err, "open-output-fail");
        cleanup();
        return false;
    }
    opened_ = true;
    return true;
}

bool FfmpegEncoder::init_encoder(int w, int h, int fps, std::string* err) {
    struct Opt { std::string name; bool hw; };
    std::vector<Opt> candidates;
    if (cfg_.codec == "auto") {
        // auto：hw_accel ∈ {Auto, Cuda} 且 nvenc 存在 → 优先 nvenc；否则（或 nvenc 打开失败）回退 libx264
        const bool want_hw = (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda) &&
                             avcodec_find_encoder_by_name("h264_nvenc") != nullptr;
        if (want_hw) candidates.push_back({"h264_nvenc", true});
        candidates.push_back({"libx264", false});
    } else if (cfg_.codec == "h264_nvenc") {
        candidates.push_back({"h264_nvenc", true});
    } else if (cfg_.codec == "libx264") {
        candidates.push_back({"libx264", false});
    } else {
        // 其它名称（含 GStreamer 名 nvh264enc/x264enc 等）在 FFmpeg 编码器里不支持，由 H4/GStreamer 编码器接
        set_err(err, "unsupported-codec");
        return false;
    }
    for (const auto& o : candidates) {
        if (!avcodec_find_encoder_by_name(o.name.c_str())) continue;
        if (configure_encoder(o.name, o.hw, w, h, fps)) {
            used_hw_ = o.hw;
            return true;
        }
        // 候选打开失败：显式 h264_nvenc 必须报错（不静默换软编）；auto 时继续尝试下一候选（软编回退）
        if (cfg_.codec == "h264_nvenc") {
            set_err(err, "encoder-open-fail");
            return false;
        }
    }
    set_err(err, "encoder-open-fail");
    return false;
}

bool FfmpegEncoder::configure_encoder(const std::string& name, bool hw, int w, int h, int fps) {
    const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
    if (!codec) return false;
    enc_ = avcodec_alloc_context3(codec);
    if (!enc_) return false;
    enc_->width = w;
    enc_->height = h;
    enc_->time_base = {1, fps};
    enc_->framerate = {fps, 1};
    const AVPixelFormat fmt = hw ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P;
    enc_->pix_fmt = fmt;
    enc_->gop_size = cfg_.gop;
    enc_->bit_rate = static_cast<int64_t>(cfg_.bitrate_kbps) * 1000;
    enc_->max_b_frames = cfg_.max_b_frames;
    // 显式设置色彩空间为 BT.709 limited range，避免播放器误解为 BT.601 导致颜色偏差
    enc_->color_range = AVCOL_RANGE_MPEG;
    enc_->colorspace = AVCOL_SPC_BT709;
    enc_->color_primaries = AVCOL_PRI_BT709;
    enc_->color_trc = AVCOL_TRC_BT709;
    if (hw) {
        // NVENC 私有属性与 libx264 不同（无 ultrafast/zerolatency）；默认 preset="ultrafast" 是 x264 专用，
        // 直接套用会让 nvenc 报 "Undefined constant"。此处不硬设 nvenc preset/tune，留 FFmpeg 默认，
        // 避免版本相关差异；显式低延迟/预设由上层后续精细配置。
    } else {
        enc_->profile = FF_PROFILE_H264_MAIN;
        enc_->level = 41;
        av_opt_set(enc_->priv_data, "preset",
                   cfg_.preset.empty() ? "ultrafast" : cfg_.preset.c_str(), 0);
        if (cfg_.low_latency) {
            av_opt_set(enc_->priv_data, "tune", "zerolatency", 0);
        }
    }
    if (avcodec_open2(enc_, codec, nullptr) < 0) {
        avcodec_free_context(&enc_);
        enc_ = nullptr;
        return false;
    }
    dst_fmt_ = fmt;
    frame_ = av_frame_alloc();
    if (!frame_) return false;
    frame_->format = fmt;
    frame_->width = w;
    frame_->height = h;
    frame_->color_range = AVCOL_RANGE_MPEG;
    frame_->colorspace = AVCOL_SPC_BT709;
    frame_->color_primaries = AVCOL_PRI_BT709;
    frame_->color_trc = AVCOL_TRC_BT709;
    if (av_frame_get_buffer(frame_, 32) < 0) return false;
    pkt_ = av_packet_alloc();
    if (!pkt_) return false;
    return true;
}

bool FfmpegEncoder::open_output(const std::string& url) {
    // 根据 cfg.format 或 URL 选择封装/输出格式
    std::string fmt = cfg_.format;
    if (fmt.empty() || fmt == "auto") {
        if (url.find("rtsp://") == 0)      fmt = "rtsp";
        else if (url.find("rtmp://") == 0) fmt = "flv";   // RTMP 容器是 FLV
        else if (url.find("http://") == 0 || url.find("https://") == 0) fmt = "flv";
        else {
            auto pos = url.find_last_of('.');
            if (pos != std::string::npos) {
                std::string ext = url.substr(pos + 1);
                fmt = (ext == "flv" || ext == "mp4") ? ext : "mp4";
            } else {
                fmt = "mp4";
            }
        }
    }
    avformat_alloc_output_context2(&fmt_, nullptr, fmt.c_str(), url.c_str());
    if (!fmt_) return false;
    st_ = avformat_new_stream(fmt_, nullptr);
    st_->time_base = enc_->time_base;
    avcodec_parameters_from_context(st_->codecpar, enc_);
    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_->pb, url.c_str(), AVIO_FLAG_WRITE) < 0) return false;
    }
    if (avformat_write_header(fmt_, nullptr) < 0) return false;
    header_ = true;
    return true;
}

bool FfmpegEncoder::encode(const modeldeploy::vision::ImageData& image, std::string* err) {
    if (!opened_ || !enc_ || !frame_) {
        set_err(err, "not-opened");
        return false;
    }
    // 防越界：输入尺寸必须与 open 时一致，否则 sws_scale 按 w_×h_ 读取会越界
    if (image.width() != w_ || image.height() != h_) {
        set_err(err, "dimension-mismatch");
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    // 取 CPU BGR 平面（packed 单平面，stride 可能含对齐；PKG_BGR_U8 每像素 3 字节）
    auto p = image.plane(0);
    if (!p.data || p.step <= 0) {
        set_err(err, "no-cpu-plane");
        return false;
    }
    if (av_frame_make_writable(frame_) < 0) {
        set_err(err, "frame-not-writable");
        return false;
    }
    if (!sws_) {
        // 注意：此 FFmpeg 的 SWS_CS_* 用低位值，不能放入 sws_getContext 的 flags（会与算法位冲突如
        // "Exactly one scaler algorithm must be chosen"）。因此用 BILINEAR 建上下文后，
        // 再通过 sws_setColorspaceDetails 显式把目标 YUV 矩阵设为 BT.709 limited，使其与
        // 编码器/帧元数据声明的 BT.709 一致——否则 RGB→YUV 默认走 BT.601，彩色内容会偏色
        // （纯灰测试掩盖了这一点）。
        sws_ = sws_getContext(w_, h_, AV_PIX_FMT_BGR24, w_, h_, dst_fmt_,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_) {
            set_err(err, "sws-init-fail");
            return false;
        }
        const int* bt709 = sws_getCoefficients(SWS_CS_ITU709);
        const int* def = sws_getCoefficients(SWS_CS_DEFAULT);
        sws_setColorspaceDetails(sws_, def, 1, bt709, 0, 0, 1 << 16, 1 << 16);
    }
    const uint8_t* src[1] = {p.data};
    int src_stride[1] = {p.step};
    sws_scale(sws_, src, src_stride, 0, h_, frame_->data, frame_->linesize);

    frame_->pts = pts_++;
    stats_.frames_in++;
    if (avcodec_send_frame(enc_, frame_) < 0) {
        set_err(err, "send-frame-fail");
        return false;
    }
    while (avcodec_receive_packet(enc_, pkt_) == 0) {
        if (st_) {
            av_packet_rescale_ts(pkt_, enc_->time_base, st_->time_base);
            pkt_->stream_index = st_->index;
        }
        if (fmt_ && av_interleaved_write_frame(fmt_, pkt_) < 0) {
            av_packet_unref(pkt_);
            set_err(err, "mux-write-fail");
            return false;
        }
        stats_.frames_out++;
        av_packet_unref(pkt_);
    }
    auto t1 = std::chrono::steady_clock::now();
    stats_.avg_encode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool FfmpegEncoder::encode_from_gpu_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                                         std::string* err) {
    (void)d_y;
    (void)d_uv;
    (void)w;
    (void)h;
    set_err(err, "not-implemented-yet");  // GPU 路径归 Phase 2 硬件任务
    return false;
}

bool FfmpegEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    // Phase1 最小实现：无独立异步线程，退化为同步编码
    if (!opened_) return false;
    return encode(image, nullptr);
}

bool FfmpegEncoder::start_async(std::string* err) {
    if (!opened_) {
        set_err(err, "not-opened");
        return false;
    }
    return true;  // Phase1 无后台线程，状态本就就绪
}

void FfmpegEncoder::stop_async() {
    // Phase1 无后台线程，空实现
}

bool FfmpegEncoder::has_permanently_failed() const {
    // Phase1 不跟踪永久失败：重连/永久判定留运行时加固任务
    return false;
}

VideoStats& FfmpegEncoder::stats() { return stats_; }

std::string FfmpegEncoder::last_error() const { return err_; }

void FfmpegEncoder::close() {
    // 只有成功写入过 header 才做完整刷写和 finalize，否则直接释放资源
    if (enc_ && header_) {
        avcodec_send_frame(enc_, nullptr);
        AVPacket* pkt = av_packet_alloc();
        while (avcodec_receive_packet(enc_, pkt) == 0) {
            if (st_) {
                av_packet_rescale_ts(pkt, enc_->time_base, st_->time_base);
                pkt->stream_index = st_->index;
            }
            if (fmt_) av_interleaved_write_frame(fmt_, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }
    cleanup();
}

void FfmpegEncoder::cleanup() {
    if (sws_) sws_freeContext(sws_);
    sws_ = nullptr;
    if (frame_) av_frame_free(&frame_);
    frame_ = nullptr;
    if (pkt_) av_packet_free(&pkt_);
    pkt_ = nullptr;
    if (fmt_) {
        if (header_) av_write_trailer(fmt_);
        if (fmt_->pb && !(fmt_->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_->pb);
        avformat_free_context(fmt_);
    }
    fmt_ = nullptr;
    st_ = nullptr;
    if (enc_) avcodec_free_context(&enc_);
    enc_ = nullptr;
    header_ = false;
    opened_ = false;
    pts_ = 0;
}

void FfmpegEncoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

} // namespace modeldeploy::video
