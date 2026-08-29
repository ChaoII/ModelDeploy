#include "csrc/video/backend/gst_encoder.h"

// ENABLE_GSTREAMER=OFF 时整个翻译单元编译为空（不引用任何 GStreamer 符号）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <chrono>
#include <cstring>
#endif

namespace modeldeploy::video {

#ifdef ENABLE_GSTREAMER

// GPU 直编：把设备 NV12 指针包装为 GStreamer CUDA memory（CUDAMemory）喂给 nvh264enc。
// gst/cuda/gstcudamemory.h 链会夹带 cudaGL.h（MSVC 上与 GL/gl.h 冲突），故只手工声明用到的
// 几个 gstcuda 符号 + 引用 cuda.h（CUdeviceptr/CUcontext），避开冲突头。
#ifdef HAVE_GSTCUDA
#include <cuda.h>
extern "C" {
typedef struct _GstCudaContext GstCudaContext;
typedef struct _GstCudaStream GstCudaStream;
typedef struct _GstCudaAllocator GstCudaAllocator;
GType gst_cuda_context_get_type(void);
GstCudaContext* gst_cuda_context_new(guint device_id);
GType gst_cuda_allocator_get_type(void);
GstMemory* gst_cuda_allocator_alloc_wrapped(GstCudaAllocator* allocator, GstCudaContext* context,
                                            GstCudaStream* stream, const GstVideoInfo* info,
                                            CUdeviceptr dev_ptr, gpointer user_data,
                                            GDestroyNotify notify);
#define GST_CUDA_ALLOCATOR(obj) ((GstCudaAllocator*)(obj))
}
#endif // HAVE_GSTCUDA

using modeldeploy::vision::ImageData;

namespace {
std::atomic<bool> g_gst_initialized{false};

void md_gst_init_once() {
    if (g_gst_initialized.load()) return;
    static std::mutex init_mtx;
    std::lock_guard<std::mutex> lk(init_mtx);
    if (g_gst_initialized.load()) return;
    GError* err = nullptr;
    if (gst_init_check(nullptr, nullptr, &err)) g_gst_initialized = true;
    if (err) g_error_free(err);
}
} // namespace

GstEncoder::GstEncoder(const VideoEncoderConfig& cfg) : cfg_(cfg) {}

GstEncoder::~GstEncoder() {
    close();
    teardown();
}

bool GstEncoder::runtime_available() const {
    // 任一编码元素可用即视为后端可用：软编 x264enc、nvh264enc、L4T nvv4l2h264enc、算能 bmh264enc。
    // 不同平台 gstreamer 的编码插件集不同（如 Sophgo linaro 可能无 x264enc 但有 bmh264enc）。
    if (x264_and_mux_available()) return true;
    if (nvh264enc_available() ||
        nvv4l2h264enc_available() ||
        bmh264enc_available()) return true;
    return false;
}

bool GstEncoder::x264_and_mux_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    bool ok = true;
    const char* names[] = {"appsrc", "videoconvert", "x264enc", "h264parse", "mp4mux",
                           "filesink"};
    for (const char* n : names) {
        GstElementFactory* f = gst_element_factory_find(n);
        if (!f) ok = false;
        if (f) gst_object_unref(f);
        if (!ok) break;
    }
    return ok;
}

bool GstEncoder::nvh264enc_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    // 与 H0 probe_gstreamer_hw() 同款检查（gst_element_factory_find("nvh264enc")），
    // 仅此一次探测即可，无需在 gst_encoder 内重复自探整链。
    GstElementFactory* f = gst_element_factory_find("nvh264enc");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

#ifdef ENABLE_VAAPI
bool GstEncoder::vaapih264enc_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    GstElementFactory* f = gst_element_factory_find("vaapih264enc");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}
#endif

bool GstEncoder::nvv4l2h264enc_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    // Jetson L4T 的 nvv4l2h264enc（gst-nvvideo4linux2）。桌面 gst-plugins-bad 无此插件名，
    // 故桌面/CUDA 场景仍走 nvh264enc；本探测用于 L4T 上 CPU 主机帧→V4L2 硬编码。
    GstElementFactory* f = gst_element_factory_find("nvv4l2h264enc");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

bool GstEncoder::bmh264enc_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    // 算能 SOPHGO BM H264 硬件编码器插件（sophon-gstreamer bmcodec 库，运行时须
    // GST_PLUGIN_PATH 指向 /opt/sophon/sophon-gstreamer_*/lib）。桌面 gstreamer 无此插件。
    GstElementFactory* f = gst_element_factory_find("bmh264enc");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

// 决议本次会话的编码元素：
//   codec=="nvh264enc" → 硬编 nvcodec（不可用则报错 no-nvh264enc）
//   codec=="nvv4l2h264enc" → Jetson L4T V4L2 硬编（不可用则报错 no-nvv4l2h264enc）
//   codec=="bmh264enc" → 算能 SOPHGO BM 硬编（不可用则报错 no-bmh264enc）
//   codec=="vaapih264enc" → VAAPI 硬编（不可用则报错 no-vaapih264enc；仅 ENABLE_VAAPI 编译时支持）
//   codec=="x264enc"/空 → 软编 x264enc
//   codec=="auto" → hw_accel∈{Auto,Cuda} 且 nvh264enc 存在则 nv 硬编（优先，支持 GPU-direct）；
//     否则 hw_accel∈{Auto,Cuda} 且 nvv4l2h264enc 存在则 L4T V4L2 硬编（CPU 帧）；否则 hw_accel∈{Auto,Vaapi}
//     且 vaapih264enc 存在则 VAAPI 硬编；否则回退软编 x264enc
//   Sophgo：GStreamer 无对应编码插件（未实现/未验证），显式 Sophgo 走 unsupported-codec fail-closed。
//   其余名称（libx264 / h264_nvenc 等 GStreamer 不支持的）→ unsupported-codec。
int GstEncoder::resolve_encoder(std::string* err) {
    const std::string& codec = cfg_.codec;
    int choice = -1;
    if (codec == "nvh264enc") {
        choice = 1;
    } else if (codec == "nvv4l2h264enc") {
        choice = 3;
    } else if (codec == "bmh264enc") {
        choice = 4;
    } else if (codec == "x264enc" || codec.empty()) {
        choice = 0;
#ifdef ENABLE_VAAPI
    } else if (codec == "vaapih264enc") {
        choice = 2;
#endif
    } else if (codec == "auto") {
        const bool want_nv = (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda);
        if (want_nv && nvh264enc_available()) {
            choice = 1;
        } else if (want_nv && nvv4l2h264enc_available()) {
            choice = 3;
#ifdef ENABLE_VAAPI
        } else if ((cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Vaapi) &&
                   vaapih264enc_available()) {
            choice = 2;
#endif
        } else {
            choice = 0;
        }
    } else {
        set_err(err, "unsupported-codec");
        return -1;
    }
    if (choice == 1 && !nvh264enc_available()) {
        set_err(err, "no-nvh264enc");
        return -1;
    }
    if (choice == 3 && !nvv4l2h264enc_available()) {
        set_err(err, "no-nvv4l2h264enc");
        return -1;
    }
    if (choice == 4 && !bmh264enc_available()) {
        set_err(err, "no-bmh264enc");
        return -1;
    }
#ifdef ENABLE_VAAPI
    if (choice == 2 && !vaapih264enc_available()) {
        set_err(err, "no-vaapih264enc");
        return -1;
    }
#endif
    if (choice == 0 && !x264_and_mux_available()) {
        set_err(err, "no-x264enc");
        return -1;
    }
    // GPU 直编（gpu_direct_input）只能走 nvh264enc 的 CUDA memory 设备路径；
    // L4T nvv4l2h264enc（choice 3）走 V4L2，不接 CUDAMemory，故 gpu_direct 时禁用它。
    if (cfg_.gpu_direct_input && choice != 1) {
        set_err(err, "gpu-direct-needs-nvh264enc");
        return -1;
    }
    encoder_is_nv_ = (choice == 1);
    encoder_is_l4t_ = (choice == 3);
    encoder_is_bm_ = (choice == 4);
#ifdef ENABLE_VAAPI
    encoder_is_vaapi_ = (choice == 2);
#endif
    return choice;
}

bool GstEncoder::open(const std::string& url, int w, int h, int src_fps,
                      const VideoEncoderConfig& c, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    teardown();
    cfg_ = c;
    w_ = w;
    h_ = h;
    // fps==0 视为自动：优先取配置，其次取源帧率，最后兜底 25
    if (cfg_.fps <= 0) cfg_.fps = (src_fps > 0) ? src_fps : 25;
    if (cfg_.gop <= 0) cfg_.gop = cfg_.fps * 2;
    fps_ = cfg_.fps;
    if (w <= 0 || h <= 0) {
        set_err(err, "invalid-dimension");
        return false;
    }
    // Phase1 软编基线仅支持 mp4 容器（mp4mux）；rtsp/rtmp 容器策划留扩展
    if (cfg_.format != "auto" && cfg_.format != "mp4") {
        set_err(err, "format-not-supported");
        return false;
    }
    md_gst_init_once();
    if (!g_gst_initialized.load()) {
        set_err(err, "gst-init-fail");
        return false;
    }
    const int enc = resolve_encoder(err);
    if (enc < 0) {
        teardown();
        return false;
    }
    build_pipeline(url, w, h, cfg_.fps, enc);
    if (!pipeline_ || !appsrc_) {
        set_err(err, "parse-launch-fail");
        teardown();
        return false;
    }
    if (!start_pipeline(err)) {
        teardown();
        return false;
    }
    opened_ = true;
    return true;
}

void GstEncoder::build_pipeline(const std::string& url, int w, int h, int fps, int enc) {
    // BGR 经 videoconvert 转 I420/NV12 供编码器；mp4mux 需在源 EOS 后才写出 moov
    std::string encoder_part;
    if (enc == 1) {
        // nvh264enc：bitrate 单位是 kbit/sec（与 cfg_.bitrate_kbps 一致，勿当 bps）；
        // GOP 属性名是 gop-size（非 x264 的 key-int-max）；preset/tune 为枚举且版本差异大，
        // 不套 x264 的 ultrafast/zerolatency 名称（会报 Undefined constant），交给 nvcodec 默认。
        encoder_part = " nvh264enc bitrate=" + std::to_string(cfg_.bitrate_kbps) +
                       " gop-size=" + std::to_string(cfg_.gop) +
                       (cfg_.low_latency ? " zerolatency=true" : "");
    } else if (enc == 3) {
        // Jetson L4T nvv4l2h264enc：bitrate 单位 kbit/sec（VERIFY：随 L4T 版本可能为 bps）、
        // GOP 关键帧间隔属性名是 iframeinterval；经 videoconvert 由 CPU 主机帧（BGR→NV12）喂入。
        encoder_part = " nvv4l2h264enc bitrate=" + std::to_string(cfg_.bitrate_kbps) +
                       " iframeinterval=" + std::to_string(cfg_.gop) +
                       (cfg_.low_latency ? " control-rate=2" : "");
    } else if (enc == 4) {
        // 算能 SOPHGO BM H264 硬编（bmh264enc）：bps 单位是 bit/sec（非 kbit），GOP 属性名 gop。
        // 经 videoconvert 由 CPU 主机帧（BGR→NV12）喂入。底层 BM VPU 硬件编码。
        encoder_part = " bmh264enc bps=" + std::to_string(cfg_.bitrate_kbps * 1000) +
                       " gop=" + std::to_string(cfg_.gop);
#ifdef ENABLE_VAAPI
    } else if (enc == 2) {
        // vaapih264enc：bitrate 单位 kbit/sec；GOP 关键帧间隔属性名因 gst-vaapi 版本而异
        // （key-int-max / gop-size），此处用 key-int-max，真机联调时按实际插件核对（VERIFY）。
        // 未在本机验证（需 Linux VAAPI + gst-vaapi）。
        encoder_part = " vaapih264enc bitrate=" + std::to_string(cfg_.bitrate_kbps) +
                       " key-int-max=" + std::to_string(cfg_.gop) +
                       (cfg_.low_latency ? " low-latency=true" : "");
#endif
    } else {
        encoder_part = " x264enc bitrate=" + std::to_string(cfg_.bitrate_kbps) +
                       " speed-preset=" + (cfg_.preset.empty() ? "ultrafast" : cfg_.preset) +
                       " tune=zerolatency key-int-max=" + std::to_string(cfg_.gop);
    }
    // GPU 直编（gpu_direct_input）：设备 NV12 CUDA memory 直接进 nvh264enc，无需 videoconvert
    const bool gpu_direct = cfg_.gpu_direct_input && (enc == 1);
    std::string launch;
    if (gpu_direct) {
        launch = "appsrc name=src format=time "
                 "! video/x-raw(memory:CUDAMemory),format=NV12 !" + encoder_part +
                 " ! h264parse ! mp4mux ! filesink location=\"" + url + "\"";
    } else {
        launch = "appsrc name=src format=time "
                 "! videoconvert !" + encoder_part +
                 " ! h264parse ! mp4mux ! filesink location=\"" + url + "\"";
    }
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (gerr) g_error_free(gerr);
    if (!pipeline_) return;
    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (!appsrc_) return;

    GstCaps* caps;
    if (gpu_direct) {
        caps = gst_caps_from_string(
            ("video/x-raw(memory:CUDAMemory),format=NV12,width=" + std::to_string(w) +
             ",height=" + std::to_string(h) + ",framerate=" + std::to_string(fps) + "/1")
                .c_str());
    } else {
        caps = gst_caps_new_simple(
            "video/x-raw", "format", G_TYPE_STRING, "BGR", "width", G_TYPE_INT, w, "height",
            G_TYPE_INT, h, "framerate", GST_TYPE_FRACTION, fps, 1, nullptr);
    }
    gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps);
    gst_caps_unref(caps);

    gst_app_src_set_stream_type(GST_APP_SRC(appsrc_), GST_APP_STREAM_TYPE_STREAM);
    g_object_set(G_OBJECT(appsrc_), "format", GST_FORMAT_TIME, nullptr);
    // 缓冲/背压策略：max-buffers + max-bytes 设上限；leaky 保持默认 none →
    // 队列满时 gst_app_src_push_buffer 阻塞推帧（天然背压，不丢帧，保证回读帧数精确）。
    gst_app_src_set_max_buffers(GST_APP_SRC(appsrc_), (guint)cfg_.async_queue_size);
    gst_app_src_set_max_bytes(GST_APP_SRC(appsrc_), (guint64)w * h * 3 / 2 * cfg_.async_queue_size);
}

bool GstEncoder::start_pipeline(std::string* err) {
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "play-fail");
        return false;
    }
    bus_ = gst_element_get_bus(pipeline_);
    return true;
}

bool GstEncoder::encode(const VideoFrame& frame, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_ || !pipeline_ || !appsrc_) {
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
            if (!cfg_.gpu_direct_input || !encoder_is_nv_) {
                set_err(err, "not-gpu-direct");
                return false;
            }
            return encode_gpu(image, frame.pts_ms, err);
        case modeldeploy::Device::CPU:
            if (cfg_.gpu_direct_input && encoder_is_nv_) {
                set_err(err, "cpu-encode-not-in-gpu-direct");
                return false;
            }
            return encode_cpu(image, frame.pts_ms, err);
        case modeldeploy::Device::TPU:
            set_err(err, "device-tpu-unavailable");
            return false;
        default:
            set_err(err, "unsupported-device");
            return false;
    }
}

bool GstEncoder::encode_cpu(const modeldeploy::vision::ImageData& image, uint64_t pts_ms,
                            std::string* err) {
    if (!opened_ || !pipeline_ || !appsrc_) {
        set_err(err, "not-opened");
        return false;
    }
    // 防越界：输入尺寸必须与 open 时一致，否则按 w_×h_ 读取会越界
    if (image.width() != w_ || image.height() != h_) {
        set_err(err, "dimension-mismatch");
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto p = image.plane(0);
    if (!p.data || p.step <= 0) {
        set_err(err, "no-cpu-plane");
        return false;
    }
    size_t row_bytes = (size_t)w_ * 3;  // PKG_BGR_U8 每像素 3 字节
    GstBuffer* buf = gst_buffer_new_and_alloc(row_bytes * h_);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    const uint8_t* src = p.data;
    uint8_t* dst = map.data;
    if (p.step == (int)row_bytes) {
        memcpy(dst, src, row_bytes * h_);
    } else {
        // 输入 stride 含对齐时逐行拷贝为连续打包 BGR
        for (int y = 0; y < h_; ++y)
            memcpy(dst + (size_t)y * row_bytes, src + (size_t)y * p.step, row_bytes);
    }
    gst_buffer_unmap(buf, &map);
    if (pts_ms != 0) GST_BUFFER_PTS(buf) = pts_ms * (GST_SECOND / 1000);
    else GST_BUFFER_PTS(buf) = (pts_ * GST_SECOND) / fps_;
    GST_BUFFER_DURATION(buf) = GST_SECOND / fps_;
    if (pts_ms == 0) pts_++;
    stats_.frames_in++;

    // push 默认阻塞：内部队列满时等下游消费，形成端到端背压
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buf);
    if (ret != GST_FLOW_OK) {
        set_err(err, "push-fail");
        return false;
    }
    stats_.frames_out++;
    auto t1 = std::chrono::steady_clock::now();
    encode_avg_sum_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.avg_encode_ms = stats_.frames_in > 0 ? encode_avg_sum_ / static_cast<double>(stats_.frames_in) : 0.0;
    return true;
}

bool GstEncoder::encode_gpu(const modeldeploy::vision::ImageData& image, uint64_t pts_ms,
                            std::string* err) {
#ifdef HAVE_GSTCUDA
    if (!opened_ || !pipeline_ || !appsrc_ || !encoder_is_nv_) {
        set_err(err, "not-opened");
        return false;
    }
    if (!cfg_.gpu_direct_input) {
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
    // 会话内 gstcuda 上下文与 allocator（同一 GPU）——成员持有、teardown 释放，避免进程级状态污染
    if (!cuda_ctx_) {
        cuda_ctx_ = gst_cuda_context_new(0);
        if (!cuda_ctx_) {
            set_err(err, "cuda-ctx-fail");
            return false;
        }
    }
    if (!cuda_alloc_) {
        cuda_alloc_ = GST_CUDA_ALLOCATOR(g_object_new(gst_cuda_allocator_get_type(), NULL));
        if (!cuda_alloc_) {
            set_err(err, "cuda-alloc-fail");
            return false;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    // 紧凑连续设备 NV12（单块，base=d_y）→ GStreamer CUDA memory（memory:CUDAMemory）
    GstVideoInfo vi;
    gst_video_info_init(&vi);
    gst_video_info_set_format(&vi, GST_VIDEO_FORMAT_NV12, w_, h_);
    vi.fps_n = fps_;
    vi.fps_d = 1;
    GstMemory* mem = gst_cuda_allocator_alloc_wrapped(cuda_alloc_, cuda_ctx_, nullptr, &vi,
                                                      (CUdeviceptr)d_y, nullptr, nullptr);
    if (!mem) {
        set_err(err, "cuda-wrap-fail");
        return false;
    }
    GstBuffer* buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    if (pts_ms != 0) GST_BUFFER_PTS(buf) = pts_ms * (GST_SECOND / 1000);
    else GST_BUFFER_PTS(buf) = (pts_ * GST_SECOND) / fps_;
    GST_BUFFER_DURATION(buf) = GST_SECOND / fps_;
    if (pts_ms == 0) pts_++;
    stats_.frames_in++;

    // push 默认阻塞：队列满时等下游消费（端到端背压）
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buf);
    if (ret != GST_FLOW_OK) {
        set_err(err, "push-fail");
        return false;
    }
    stats_.frames_out++;
    auto t1 = std::chrono::steady_clock::now();
    encode_avg_sum_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.avg_encode_ms = stats_.frames_in > 0 ? encode_avg_sum_ / static_cast<double>(stats_.frames_in) : 0.0;
    return true;
#else
    (void)image;
    set_err(err, "device-encode-unsupported");
    return false;
#endif
}

bool GstEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    // Phase1 最小实现：无独立异步线程，退化为同步编码
    return encode(VideoFrame{image, 0}, nullptr);
}

bool GstEncoder::start_async(std::string* err) {
    if (!opened_) {
        set_err(err, "not-opened");
        return false;
    }
    return true;  // Phase1 无后台线程，状态本就就绪
}

void GstEncoder::stop_async() {
    // Phase1 无后台线程，空实现
}

bool GstEncoder::has_permanently_failed() const {
    // Phase1 不跟踪永久失败
    return false;
}

VideoStats& GstEncoder::stats() { return stats_; }

std::string GstEncoder::last_error() const { return err_; }

void GstEncoder::close() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_ && !pipeline_) return;
    // 关键时序：先向 appsrc 刷 EOS，等 mp4mux 收到 EOS 并收尾写出 moov，再停管道。
    // 若直接 set_state(NULL) 会导致 mp4mux 未 finalize，产物缺 moov 无法回读。
    if (appsrc_) gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
    wait_eos_and_stop();
}

void GstEncoder::wait_eos_and_stop() {
    if (!pipeline_) return;
    auto begin = std::chrono::steady_clock::now();
    while (bus_) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus_, 100 * GST_MSECOND, (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        if (!msg) {
            // 轮询无消息：超时兜底退出，避免无限等待
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - begin)
                    .count() > 20000)
                break;
            continue;
        }
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* e = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &e, &dbg);
            if (e) g_error_free(e);
            if (dbg) g_free(dbg);
            gst_message_unref(msg);
            break;
        }
        // EOS：mp4mux 已收尾，moov 已写出
        gst_message_unref(msg);
        break;
    }
    teardown();
}

void GstEncoder::teardown() {
    if (bus_) { gst_object_unref(bus_); bus_ = nullptr; }
    if (appsrc_) { gst_object_unref(appsrc_); appsrc_ = nullptr; }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
#ifdef HAVE_GSTCUDA
    // 释放会话 GPU 直编状态：不跨会话保留 CUDA 上下文，避免污染同进程后续管道
    if (cuda_alloc_) { gst_object_unref(cuda_alloc_); cuda_alloc_ = nullptr; }
    if (cuda_ctx_) { gst_object_unref(cuda_ctx_); cuda_ctx_ = nullptr; }
#endif
    w_ = 0;
    h_ = 0;
    fps_ = 0;
    pts_ = 0;
    opened_ = false;
    encoder_is_nv_ = false;
    encoder_is_l4t_ = false;
    encoder_is_bm_ = false;
#ifdef ENABLE_VAAPI
    encoder_is_vaapi_ = false;
#endif
}

void GstEncoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

#endif // ENABLE_GSTREAMER
} // namespace modeldeploy::video
