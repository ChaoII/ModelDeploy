#include "csrc/video/backend/gst_encoder.h"

// ENABLE_GSTREAMER=OFF 时整个翻译单元编译为空（不引用任何 GStreamer 符号）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <chrono>
#include <cstring>
#endif

namespace modeldeploy::video {

#ifdef ENABLE_GSTREAMER

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

bool GstEncoder::runtime_available() const { return gstreamer_x264_available(); }

bool GstEncoder::gstreamer_x264_available() {
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

// 决议本次会话的编码元素：
//   codec=="nvh264enc" → 硬编（不可用则报错 no-nvh264enc）
//   codec=="x264enc"/空 → 软编 x264enc
//   codec=="auto" → hw_accel∈{Auto,Cuda} 且 nvh264enc 存在则硬编，否则回退软编
//   VAAPI/Sophgo：本期仅作为可接受配置值 + 由 H0 探测上报（有 vaapih264enc/h264_vaapi 会列出），
//   未实现实际帧路径；auto 下显式 HwAccel::Vaapi/Sophgo 一律回退软编 x264enc。
//   其余名称（libx264 / h264_nvenc / vaapih264enc 等 GStreamer 不支持的）→ unsupported-codec。
int GstEncoder::resolve_encoder(std::string* err) {
    const std::string& codec = cfg_.codec;
    int choice = -1;
    if (codec == "nvh264enc") {
        choice = 1;
    } else if (codec == "x264enc" || codec.empty()) {
        choice = 0;
    } else if (codec == "auto") {
        const bool want_hw = (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda);
        choice = (want_hw && nvh264enc_available()) ? 1 : 0;
    } else {
        set_err(err, "unsupported-codec");
        return -1;
    }
    if (choice == 1 && !nvh264enc_available()) {
        set_err(err, "no-nvh264enc");
        return -1;
    }
    if (choice == 0 && !gstreamer_x264_available()) {
        set_err(err, "no-x264enc");
        return -1;
    }
    encoder_is_nv_ = (choice == 1);
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
    } else {
        encoder_part = " x264enc bitrate=" + std::to_string(cfg_.bitrate_kbps) +
                       " speed-preset=" + (cfg_.preset.empty() ? "ultrafast" : cfg_.preset) +
                       " tune=zerolatency key-int-max=" + std::to_string(cfg_.gop);
    }
    std::string launch =
        "appsrc name=src format=time "
        "! videoconvert !" + encoder_part +
        " ! h264parse ! mp4mux ! filesink location=\"" + url + "\"";
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (gerr) g_error_free(gerr);
    if (!pipeline_) return;
    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (!appsrc_) return;

    GstCaps* caps = gst_caps_new_simple(
        "video/x-raw", "format", G_TYPE_STRING, "BGR", "width", G_TYPE_INT, w, "height",
        G_TYPE_INT, h, "framerate", GST_TYPE_FRACTION, fps, 1, nullptr);
    gst_app_src_set_caps(GST_APP_SRC(appsrc_), caps);
    gst_caps_unref(caps);

    gst_app_src_set_stream_type(GST_APP_SRC(appsrc_), GST_APP_STREAM_TYPE_STREAM);
    g_object_set(G_OBJECT(appsrc_), "format", GST_FORMAT_TIME, nullptr);
    // 缓冲/背压策略：max-buffers + max-bytes 设上限；leaky 保持默认 none →
    // 队列满时 gst_app_src_push_buffer 阻塞推帧（天然背压，不丢帧，保证回读帧数精确）。
    gst_app_src_set_max_buffers(GST_APP_SRC(appsrc_), (guint)cfg_.async_queue_size);
    gst_app_src_set_max_bytes(GST_APP_SRC(appsrc_), (guint64)w * h * 3 * cfg_.async_queue_size);
}

bool GstEncoder::start_pipeline(std::string* err) {
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "play-fail");
        return false;
    }
    bus_ = gst_element_get_bus(pipeline_);
    return true;
}

bool GstEncoder::encode(const modeldeploy::vision::ImageData& image, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
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
    GST_BUFFER_PTS(buf) = (pts_ * GST_SECOND) / fps_;
    GST_BUFFER_DURATION(buf) = GST_SECOND / fps_;
    pts_++;
    stats_.frames_in++;

    // push 默认阻塞：内部队列满时等下游消费，形成端到端背压
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buf);
    if (ret != GST_FLOW_OK) {
        set_err(err, "push-fail");
        return false;
    }
    stats_.frames_out++;
    auto t1 = std::chrono::steady_clock::now();
    stats_.avg_encode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool GstEncoder::encode_from_gpu_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                                      std::string* err) {
    (void)d_y;
    (void)d_uv;
    (void)w;
    (void)h;
    set_err(err, "not-implemented-yet");  // GPU 路径归 Phase 2 硬件任务
    return false;
}

bool GstEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    // Phase1 最小实现：无独立异步线程，退化为同步编码
    return encode(image, nullptr);
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
    w_ = 0;
    h_ = 0;
    fps_ = 0;
    pts_ = 0;
    opened_ = false;
}

void GstEncoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

#endif // ENABLE_GSTREAMER
} // namespace modeldeploy::video
