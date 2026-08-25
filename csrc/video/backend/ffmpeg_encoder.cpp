#include "csrc/video/backend/ffmpeg_encoder.h"

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#if defined(ENABLE_VAAPI)
// VAAPI 硬件上下文头：仅在编译 VAAPI 路径时引入（依赖 libva 头，Linux 环境才有）
#include <libavutil/hwcontext_vaapi.h>
#endif
// 仅当 CUDA 驱动头可用（gstcuda 探测到 CUDA include 目录）时启用设备编解码的 D2D 路径。
// 用 CUDA 驱动 API（cuda.h / nvcuda.dll）而非 cudart 运行时，最小化链接依赖。
#ifdef MODELDEPLOY_CUDA_DRV
#include <libavutil/hwcontext_cuda.h>
#include <cuda.h>
#endif
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
    // kind: 0=软编 libx264, 1=NVENC(nvenc), 2=VAAPI(h264_vaapi)
    struct Opt { std::string name; int kind; };
    std::vector<Opt> candidates;
    if (cfg_.codec == "auto") {
        // auto：hw_accel ∈ {Auto, Cuda} 且 nvenc 存在 → 优先 nvenc；hw_accel ∈ {Auto,Vaapi} 且
        // VAAPI 编译开启 → 尝试 h264_vaapi；否则（或打开失败）回退 libx264
        const bool want_nvenc = (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda) &&
                                avcodec_find_encoder_by_name("h264_nvenc") != nullptr;
        if (want_nvenc) candidates.push_back({"h264_nvenc", 1});
#ifdef ENABLE_VAAPI
        if (!want_nvenc &&
            (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Vaapi) &&
            avcodec_find_encoder_by_name("h264_vaapi") != nullptr)
            candidates.push_back({"h264_vaapi", 2});
#endif
        candidates.push_back({"libx264", 0});
    } else if (cfg_.codec == "h264_nvenc") {
        candidates.push_back({"h264_nvenc", 1});
    } else if (cfg_.codec == "libx264") {
        candidates.push_back({"libx264", 0});
#ifdef ENABLE_VAAPI
    } else if (cfg_.codec == "vaapih264enc" || cfg_.codec == "h264_vaapi") {
        // VAAPI 编码器（GStreamer 名 vaapih264enc 与 FFmpeg 名 h264_vaapi 都映射到 FFmpeg h264_vaapi）
        candidates.push_back({"h264_vaapi", 2});
#endif
    } else {
        // 其它名称（含 GStreamer 名 nvh264enc/x264enc 等；VAAPI 未编译时的 vaapih264enc 等）
        // 在 FFmpeg 编码器里不支持，由 GStreamer 编码器接；VAAPI 未编译则明确 unsupported-codec。
        set_err(err, "unsupported-codec");
        return false;
    }
    for (const auto& o : candidates) {
        if (!avcodec_find_encoder_by_name(o.name.c_str())) continue;
        if (configure_encoder(o.name, o.kind, w, h, fps)) {
            used_hw_ = (o.kind != 0);
#ifdef ENABLE_VAAPI
            vaapi_active_ = (o.kind == 2);
#endif
            return true;
        }
        // 候选打开失败：显式 nvenc/vaapi 必须报错（不静默换软编）；auto 时继续尝试下一候选（软编回退）
        if (cfg_.codec == "h264_nvenc" || cfg_.codec == "vaapih264enc" ||
            cfg_.codec == "h264_vaapi") {
            set_err(err, "encoder-open-fail");
            return false;
        }
    }
    set_err(err, "encoder-open-fail");
    return false;
}

bool FfmpegEncoder::configure_encoder(const std::string& name, int kind, int w, int h, int fps) {
    const AVCodec* codec = avcodec_find_encoder_by_name(name.c_str());
    if (!codec) return false;
    enc_ = avcodec_alloc_context3(codec);
    if (!enc_) return false;
    enc_->width = w;
    enc_->height = h;
    enc_->time_base = {1, fps};
    enc_->framerate = {fps, 1};
    const bool hw = (kind != 0);
#ifdef ENABLE_VAAPI
    const bool vaapi = (kind == 2);
#else
    const bool vaapi = false;
#endif
    // GPU 直接编码：enc 输入 pix_fmt 为 CUDA（吃设备 NV12 hw frame）；
    // 软编 libx264 用 YUV420P，普通 nvenc（CPU NV12 上传）与 VAAPI 均用 NV12。
    const bool gpu_direct = (kind == 1) && cfg_.gpu_direct_input;
    const AVPixelFormat enc_fmt = kind == 1 ? (gpu_direct ? AV_PIX_FMT_CUDA : AV_PIX_FMT_NV12)
                                            : (vaapi ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P);
    enc_->pix_fmt = enc_fmt;
    enc_->gop_size = cfg_.gop;
    enc_->bit_rate = static_cast<int64_t>(cfg_.bitrate_kbps) * 1000;
    enc_->max_b_frames = cfg_.max_b_frames;
    // 显式设置色彩空间为 BT.709 limited range，避免播放器误解为 BT.601 导致颜色偏差
    enc_->color_range = AVCOL_RANGE_MPEG;
    enc_->colorspace = AVCOL_SPC_BT709;
    enc_->color_primaries = AVCOL_PRI_BT709;
    enc_->color_trc = AVCOL_TRC_BT709;
    if (hw) {
        if (kind == 1) {
            // NVENC：GPU 直编（gpu_direct）须在 open 前挂上 CUDA hw_frames_ctx，nvenc 依据
            // avctx->hw_frames_ctx 取 CUDA 设备上下文并注册输入资源。普通 nvenc（CPU NV12 上传）
            // 不设 hw ctx，由 FFmpeg 自建 CUDA 设备。
            if (gpu_direct) {
                if (!setup_cuda_hw_frames(w, h)) {
                    avcodec_free_context(&enc_);
                    enc_ = nullptr;
                    return false;
                }
                enc_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
                if (!enc_->hw_frames_ctx) {
                    avcodec_free_context(&enc_);
                    enc_ = nullptr;
                    return false;
                }
            }
        } else {  // VAAPI：挂 VAAPI hw 帧上下文（format=VAAPI, sw_format=NV12）
#ifdef ENABLE_VAAPI
            if (!setup_vaapi_hw_frames(w, h)) {
                avcodec_free_context(&enc_);
                enc_ = nullptr;
                return false;
            }
            enc_->hw_frames_ctx = av_buffer_ref(vaapi_hw_frames_);
            if (!enc_->hw_frames_ctx) {
                avcodec_free_context(&enc_);
                enc_ = nullptr;
                return false;
            }
#else
            avcodec_free_context(&enc_);
            enc_ = nullptr;
            return false;
#endif
        }
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
    // sws 输出的 CPU 中间/帧格式：nvenc（含 GPU 直编）与 VAAPI 输入 CPU 平面为 NV12，软编为 YUV420P
    const AVPixelFormat cpu_fmt = hw ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P;
    dst_fmt_ = cpu_fmt;
    frame_ = av_frame_alloc();
    if (!frame_) return false;
    frame_->format = cpu_fmt;
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

// GPU 直接编码：创建 CUDA 设备上下文 + CUDA hw帧上下文（format=CUDA, sw_format=NV12）。
bool FfmpegEncoder::setup_cuda_hw_frames(int w, int h) {
    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) != 0)
        return false;
    AVBufferRef* fr = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!fr) return false;
    AVHWFramesContext* fc = (AVHWFramesContext*)fr->data;
    fc->format = AV_PIX_FMT_CUDA;
    fc->sw_format = AV_PIX_FMT_NV12;
    fc->width = w;
    fc->height = h;
    if (av_hwframe_ctx_init(fr) < 0) {
        av_buffer_unref(&fr);
        return false;
    }
    hw_frames_ctx_ = fr;
    return true;
}

#ifdef ENABLE_VAAPI
// VAAPI：创建设备上下文 + NV12 hw帧上下文（format=VAAPI, sw_format=NV12），供 h264_vaapi 输入。
// 未在本机验证（需 Linux VAAPI + libva）。
bool FfmpegEncoder::setup_vaapi_hw_frames(int w, int h) {
    if (av_hwdevice_ctx_create(&vaapi_hw_ctx_, AV_HWDEVICE_TYPE_VAAPI, nullptr, nullptr, 0) != 0)
        return false;
    AVBufferRef* fr = av_hwframe_ctx_alloc(vaapi_hw_ctx_);
    if (!fr) return false;
    AVHWFramesContext* fc = (AVHWFramesContext*)fr->data;
    fc->format = AV_PIX_FMT_VAAPI;
    fc->sw_format = AV_PIX_FMT_NV12;
    fc->width = w;
    fc->height = h;
    fc->initial_pool_size = 12;
    if (av_hwframe_ctx_init(fr) < 0) {
        av_buffer_unref(&fr);
        return false;
    }
    vaapi_hw_frames_ = fr;
    return true;
}

// 把 CPU NV12 帧（frame_）上传为 VAAPI hw 帧并送入编码器，做收包写出循环。
bool FfmpegEncoder::send_vaapi_frame(std::string* err) {
    if (!enc_ || !frame_ || !pkt_) {
        set_err(err, "not-opened");
        return false;
    }
    AVFrame* hw = av_frame_alloc();
    if (!hw) {
        set_err(err, "frame-alloc-fail");
        return false;
    }
    if (av_hwframe_get_buffer(vaapi_hw_frames_, hw, 0) < 0) {
        av_frame_free(&hw);
        set_err(err, "hwframe-alloc-fail");
        return false;
    }
    if (av_hwframe_transfer_data(hw, frame_, 0) < 0) {  // CPU NV12 → VAAPI hw 帧
        av_frame_free(&hw);
        set_err(err, "hwframe-upload-fail");
        return false;
    }
    hw->pts = pts_++;
    stats_.frames_in++;
    if (avcodec_send_frame(enc_, hw) < 0) {
        av_frame_free(&hw);
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
            av_frame_free(&hw);
            set_err(err, "mux-write-fail");
            return false;
        }
        stats_.frames_out++;
        av_packet_unref(pkt_);
    }
    av_frame_free(&hw);
    return true;
}
#endif // ENABLE_VAAPI

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

bool FfmpegEncoder::encode(const VideoFrame& frame, std::string* err) {
    if (!opened_ || !enc_ || !frame_) {
        set_err(err, "not-opened");
        return false;
    }
    const auto& image = frame.image;
    if (image.width() != w_ || image.height() != h_) {
        set_err(err, "dimension-mismatch");
        return false;
    }
    switch (image.device()) {
        case modeldeploy::Device::GPU:
            return encode_gpu(image, err);
        case modeldeploy::Device::CPU:
            return encode_cpu(image, err);
        case modeldeploy::Device::TPU:
            set_err(err, "device-tpu-unavailable");
            return false;
        default:
            set_err(err, "unsupported-device");
            return false;
    }
}

bool FfmpegEncoder::encode_cpu(const modeldeploy::vision::ImageData& image, std::string* err) {
    if (!opened_ || !enc_ || !frame_) {
        set_err(err, "not-opened");
        return false;
    }
    // GPU 直编会话编码器输入为 CUDA hw 帧，CPU BGR 直送会被误读为设备指针 → 拒绝
    if (cfg_.gpu_direct_input && used_hw_) {
        set_err(err, "cpu-encode-not-in-gpu-direct");
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

#ifdef ENABLE_VAAPI
    // VAAPI 硬编：CPU NV12 经 av_hwframe_transfer_data 上传为 VAAPI hw 帧后送入编码器
    // （send_vaapi_frame 内部维护 pts_/stats_）。未在本机验证（需 Linux VAAPI）。
    if (vaapi_active_) {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = send_vaapi_frame(err);
        auto t1 = std::chrono::steady_clock::now();
        stats_.avg_encode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ok;
    }
#endif

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

bool FfmpegEncoder::encode_gpu(const modeldeploy::vision::ImageData& image, std::string* err) {
    if (!opened_ || !enc_ || !pkt_) {
        set_err(err, "not-opened");
        return false;
    }
    // GPU 直编是独立设备会话：需 open 时以 gpu_direct_input 决议出 nvenc + CUDA hw 帧上下文
    if (!cfg_.gpu_direct_input || !used_hw_ || !hw_frames_ctx_) {
        set_err(err, "not-gpu-direct");
        return false;
    }
    if (image.width() != w_ || image.height() != h_) {
        set_err(err, "dimension-mismatch");
        return false;
    }
    auto py = image.plane(0);
    auto puv = image.plane(1);
    if (!py.data || !puv.data || py.step != w_ || puv.step != w_) {
        set_err(err, py.data && puv.data ? "device-step-mismatch" : "no-device-plane");
        return false;
    }
    const uint8_t* d_y = py.data;
    const uint8_t* d_uv = puv.data;
#ifndef MODELDEPLOY_CUDA_DRV
    (void)d_y;
    (void)d_uv;
    set_err(err, "device-encode-unsupported");
    return false;
#else
    auto t0 = std::chrono::steady_clock::now();
    // 从编码器自己的 CUDA hw帧上下文分配一帧设备内存（与 nvenc 同上下文，注册必然成功）。
    // 上层传入的设备指针可能来自其它 CUDA 上下文/驱动（如 cudaMalloc primary ctx），不能直接注册；
    // 故用 cuMemcpy2D 做一次 D2D 拷贝（device→device，含不同 pitch 的 2D 拷贝），不落主机。
    AVFrame* hw = av_frame_alloc();
    if (!hw) {
        set_err(err, "frame-alloc-fail");
        return false;
    }
    if (av_hwframe_get_buffer(hw_frames_ctx_, hw, 0) < 0) {
        av_frame_free(&hw);
        set_err(err, "hwframe-alloc-fail");
        return false;
    }
    if (!d2d_copy_nv12(d_y, d_uv, w_, h_, hw)) {
        av_frame_free(&hw);
        set_err(err, "d2d-copy-fail");
        return false;
    }
    hw->pts = pts_++;

    stats_.frames_in++;
    if (avcodec_send_frame(enc_, hw) < 0) {
        av_frame_free(&hw);
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
            av_frame_free(&hw);
            set_err(err, "mux-write-fail");
            return false;
        }
        stats_.frames_out++;
        av_packet_unref(pkt_);
    }
    av_frame_free(&hw);
    auto t1 = std::chrono::steady_clock::now();
    stats_.avg_encode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
#endif
}

// 设备→设备的 NV12 2D 拷贝（Y/UV 各一次 cuMemcpy2D）。源紧凑连续（pitch=w），
// 目标为 FFmpeg CUDA hw 帧（pitch=linesize，可能含对齐）。在当前 CUDA 上下文≠编码器上下文时
// 也成立：设备指针在整张卡上共享地址空间。返回是否全部拷贝成功。
bool FfmpegEncoder::d2d_copy_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                                  AVFrame* hw) {
#ifdef MODELDEPLOY_CUDA_DRV
    AVHWDeviceContext* dev = (AVHWDeviceContext*)hw_device_ctx_->data;
    if (!dev || !dev->hwctx) return false;
    AVCUDADeviceContext* cu = (AVCUDADeviceContext*)dev->hwctx;
    CUcontext dummy = nullptr;
    if (cuCtxPushCurrent(cu->cuda_ctx) != CUDA_SUCCESS) return false;

    CUresult r = CUDA_SUCCESS;
    const auto copy_plane = [&](CUdeviceptr dst, size_t dst_pitch, const uint8_t* src,
                                size_t src_pitch, size_t width_bytes, size_t rows) {
        CUDA_MEMCPY2D cp = {};
        cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice = (CUdeviceptr)src;
        cp.srcPitch = src_pitch;
        cp.srcXInBytes = 0;
        cp.srcY = 0;
        cp.dstDevice = dst;
        cp.dstPitch = dst_pitch;
        cp.dstXInBytes = 0;
        cp.dstY = 0;
        cp.WidthInBytes = width_bytes;
        cp.Height = rows;
        return cuMemcpy2D(&cp) == CUDA_SUCCESS;
    };
    r = copy_plane((CUdeviceptr)hw->data[0], (size_t)hw->linesize[0], d_y, (size_t)w, (size_t)w,
                   (size_t)h)
            ? CUDA_SUCCESS
            : CUDA_ERROR_UNKNOWN;
    if (r == CUDA_SUCCESS) {
        if (!copy_plane((CUdeviceptr)hw->data[1], (size_t)hw->linesize[1], d_uv, (size_t)w,
                        (size_t)w, (size_t)(h / 2)))
            r = CUDA_ERROR_UNKNOWN;
    }
    cuCtxPopCurrent(&dummy);
    return r == CUDA_SUCCESS;
#else
    (void)d_y;
    (void)d_uv;
    (void)hw;
    return false;
#endif
}

bool FfmpegEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    // Phase1 最小实现：无独立异步线程，退化为同步编码
    if (!opened_) return false;
    VideoFrame vf;
    vf.image = image;
    return encode(vf, nullptr);
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
    if (hw_frames_ctx_) av_buffer_unref(&hw_frames_ctx_);
    hw_frames_ctx_ = nullptr;
    if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
    hw_device_ctx_ = nullptr;
#ifdef ENABLE_VAAPI
    if (vaapi_hw_frames_) av_buffer_unref(&vaapi_hw_frames_);
    vaapi_hw_frames_ = nullptr;
    if (vaapi_hw_ctx_) av_buffer_unref(&vaapi_hw_ctx_);
    vaapi_hw_ctx_ = nullptr;
    vaapi_active_ = false;
#endif
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
