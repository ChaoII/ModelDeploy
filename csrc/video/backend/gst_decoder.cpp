#include "csrc/video/backend/gst_decoder.h"

// ENABLE_GSTREAMER=OFF 时整个翻译单元编译为空（不引用任何 GStreamer 符号）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <chrono>
#include <cstring>
// GStreamer CUDA 设备帧映射：gst_video_frame_map 以 GST_MAP_CUDA 把 CUDA memory 映为设备平面。
// gst/cuda/gstcudamemory.h 链会夹带 cudaGL.h（MSVC 上与 GL/gl.h 冲突），故只引用宏值，不引该头。
// GST_MAP_CUDA = GST_MAP_FLAG_LAST << 1（GStreamer ≤1.30 恒定）。
#ifdef HAVE_GSTCUDA
// GST_MAP_CUDA = GST_MAP_FLAG_LAST(1<<16) << 1 = 1<<17（GStreamer ≤1.30 恒定）
#ifndef GST_MAP_CUDA
#define GST_MAP_CUDA ((GstMapFlags)(GST_MAP_FLAG_LAST << 1))
#endif
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))
#endif
#endif
// Jetson L4T：NvBufSurface 零拷贝。nvbufsurface.h 由 cmake/gstreamer.cmake 探测
// /usr/src/jetson_multimedia_api/include 提供，仅 L4T 构建定义 HAVE_NVBUF 时引入。
#ifdef HAVE_NVBUF
#include "nvbufsurface.h"
#endif
#endif

namespace modeldeploy::video {

#ifdef ENABLE_GSTREAMER

using modeldeploy::Device;

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

GstDecoder::GstDecoder(const VideoDecoderConfig& cfg)
    : cfg_(cfg), state_(State::Idle) {}

GstDecoder::~GstDecoder() {
    cleanup();
    close_pipeline();
}

bool GstDecoder::runtime_available() const { return gstreamer_available(); }

bool GstDecoder::gstreamer_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    bool ok = true;
    const char* names[] = {"filesrc", "decodebin", "videoconvert"};
    for (const char* n : names) {
        GstElementFactory* f = gst_element_factory_find(n);
        if (!f) ok = false;
        if (f) gst_object_unref(f);
        if (!ok) break;
    }
    return ok;
}

bool GstDecoder::open(const std::string& url, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    cleanup();
    close_pipeline();
    md_gst_init_once();
    if (!g_gst_initialized.load()) {
        set_err(err, "gst-init-fail");
        state_ = State::Error;
        return false;
    }
    // Sophgo：GStreamer 无对应解码插件（未实现/未验证），fail-closed 不静默软解。
    if (cfg_.hw_accel == HwAccel::Sophgo) {
        set_err(err, "sophgo-decode-requires-sophonmw");
        state_ = State::Error;
        return false;
    }
#ifndef ENABLE_VAAPI
    // VAAPI 未编译（默认）：显式请求 Vaapi → fail-closed，不静默软解。
    if (cfg_.hw_accel == HwAccel::Vaapi) {
        set_err(err, "vaapi-unavailable");
        state_ = State::Error;
        return false;
    }
#endif
    // 设备直通：解码输出保持 CUDA 设备帧（nvh264dec → CUDA memory）
    if (cfg_.device_only && (cfg_.hw_accel == HwAccel::Auto || cfg_.hw_accel == HwAccel::Cuda)) {
#ifdef HAVE_NVBUF
        // Jetson L4T：nvv4l2decoder 硬件解码 + nvvidconv → 主机 NV12（GStreamer 标准取帧）。
        // L4T 无 CUDA 零拷贝（NvBufSurface 提取依赖私有 nvmm buffer-pool，本 SDK 不含），故 convert
        // 为主机 NV12、decode 仍硬件加速。device_only 在 L4T 亦走此路径（零拷贝不可得）。
        if (nvv4l2decoder_available()) {
            if (!build_hwdecode_pipeline_locked(url, err)) {
                state_ = State::Error;
                return false;
            }
            opened_ = true;
            state_ = State::Running;
            return true;
        }
#endif
#ifdef HAVE_GSTCUDA
        if (!build_device_pipeline_locked(url, err)) {
            state_ = State::Error;
            return false;
        }
        opened_ = true;
        state_ = State::Running;
        return true;
#else
        set_err(err, "nvcodec-not-supported");  // 未编译 gstnvcodec/CUDA 支持 → 能力不足
        state_ = State::Error;
        return false;
#endif
    }
#ifdef ENABLE_VAAPI
    // VAAPI 硬解（显式 Vaapi 或 Auto）：filesrc→h264parse→vaapih264dec→videoconvert→appsink(NV12)。
    // 显式 Vaapi 失败 → fail-closed；Auto 失败/无插件 → 落在下方软解。未在本机验证（需 Linux VAAPI）。
    if (!cfg_.device_only &&
        (cfg_.hw_accel == HwAccel::Vaapi || cfg_.hw_accel == HwAccel::Auto)) {
        if (!vaapih264dec_available()) {
            if (cfg_.hw_accel == HwAccel::Vaapi) {
                set_err(err, "vaapi-unavailable");
                state_ = State::Error;
                return false;
            }
        } else if (!build_vaapi_pipeline_locked(url, err)) {
            if (cfg_.hw_accel == HwAccel::Vaapi) {
                state_ = State::Error;
                return false;
            }
        } else {
            opened_ = true;
            state_ = State::Running;
            return true;
        }
    }
#endif
    std::string launch = "filesrc location=\"" + url +
                         "\" ! decodebin ! videoconvert "
                         "! appsink name=sink caps=\"video/x-raw,format=NV12\"";
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (!pipeline_ || gerr) {
        if (gerr) g_error_free(gerr);
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        set_err(err, "parse-launch-fail");
        state_ = State::Error;
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        set_err(err, "no-appsink");
        close_pipeline();
        state_ = State::Error;
        return false;
    }
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "play-fail");
        close_pipeline();
        state_ = State::Error;
        return false;
    }
    // 本地文件首帧很快到达：阻塞等 appsink sink pad 协商出 caps，从而 open 后即可取宽高/帧率
    query_caps_locked(5000);
    opened_ = true;
    state_ = State::Running;
    return true;
}

#ifdef HAVE_GSTCUDA
bool GstDecoder::build_device_pipeline_locked(const std::string& url, std::string* err) {
    close_pipeline();
    // 检查 nvh264dec 插件可实例化；缺失 → 能力不足（SKIP 判定前缀 nvcodec-）
    GstElementFactory* f = gst_element_factory_find("nvh264dec");
    if (!f) {
        set_err(err, "nvcodec-plugin-unavailable");
        return false;
    }
    gst_object_unref(f);
    // 显式 nvh264dec 解码 → 设备 CUDA memory，appsink 保持 memory:CUDAMemory（不做 D2H）。
    std::string launch = "filesrc location=\"" + url +
                         "\" ! h264parse ! nvh264dec "
                         "! appsink name=sink caps=\"video/x-raw(memory:CUDAMemory),format=NV12\"";
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (!pipeline_ || gerr) {
        if (gerr) g_error_free(gerr);
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        set_err(err, "parse-launch-fail");
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        set_err(err, "no-appsink");
        close_pipeline();
        return false;
    }
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "nvcodec-play-fail");
        close_pipeline();
        return false;
    }
    query_caps_locked(5000);
    device_only_active_ = true;
    return true;
}
#endif // HAVE_GSTCUDA

#ifdef HAVE_NVBUF
bool GstDecoder::nvv4l2decoder_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    // Jetson L4T V4L2 解码器（gst-nvvideo4linux2）；桌面 gst-plugins-bad 无此插件名，因此只在 L4T 命中选择。
    GstElementFactory* f = gst_element_factory_find("nvv4l2decoder");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

// Jetson L4T 硬件解码：nvv4l2decoder（真硬解）→ nvvidconv → appsink(主机 NV12)。
// L4T 无 CUDA 零拷贝，故以 nvvidconv 转为标准主机 NV12，复用下方软解 read 路径（device_only_active_ 保持 false）。
bool GstDecoder::build_hwdecode_pipeline_locked(const std::string& url, std::string* err) {
    close_pipeline();
    std::string launch = "filesrc location=\"" + url +
                         "\" ! h264parse ! nvv4l2decoder ! nvvidconv "
                         "! appsink name=sink caps=\"video/x-raw,format=NV12\"";
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (!pipeline_ || gerr) {
        if (gerr) g_error_free(gerr);
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        set_err(err, "parse-launch-fail");
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        set_err(err, "no-appsink");
        close_pipeline();
        return false;
    }
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "nvcodec-play-fail");
        close_pipeline();
        return false;
    }
    query_caps_locked(5000);
    l4t_hw_active_ = true;
    return true;
}
#endif // HAVE_NVBUF

#ifdef ENABLE_VAAPI
bool GstDecoder::vaapih264dec_available() {
    md_gst_init_once();
    if (!g_gst_initialized.load()) return false;
    GstElementFactory* f = gst_element_factory_find("vaapih264dec");
    if (!f) return false;
    gst_object_unref(f);
    return true;
}

// VAAPI 硬解码管道：显式 vaapih264dec → videoconvert → appsink(NV12)，输出主机 NV12，
// 复用软解 read 路径（device_only_active_ 保持 false）。未在本机验证（需 Linux VAAPI + gst-vaapi）。
bool GstDecoder::build_vaapi_pipeline_locked(const std::string& url, std::string* err) {
    close_pipeline();
    std::string launch = "filesrc location=\"" + url +
                         "\" ! h264parse ! vaapih264dec ! videoconvert "
                         "! appsink name=sink caps=\"video/x-raw,format=NV12\"";
    GError* gerr = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &gerr);
    if (!pipeline_ || gerr) {
        if (gerr) g_error_free(gerr);
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        set_err(err, "parse-launch-fail");
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        set_err(err, "no-appsink");
        close_pipeline();
        return false;
    }
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        set_err(err, "vaapi-play-fail");
        close_pipeline();
        return false;
    }
    query_caps_locked(5000);
    vaapi_active_ = true;
    return true;
}
#endif // ENABLE_VAAPI

bool GstDecoder::query_caps_locked(int timeout_ms) {
    if (!appsink_) return false;
    auto begin = std::chrono::steady_clock::now();
    while (true) {
        GstPad* pad = gst_element_get_static_pad(appsink_, "sink");
        if (pad) {
            GstCaps* caps = gst_pad_get_current_caps(pad);
            if (caps) {
                GstVideoInfo info;
                if (gst_video_info_from_caps(&info, caps)) {
                    if (GST_VIDEO_INFO_WIDTH(&info) > 0 && GST_VIDEO_INFO_HEIGHT(&info) > 0) {
                        w_ = GST_VIDEO_INFO_WIDTH(&info);
                        h_ = GST_VIDEO_INFO_HEIGHT(&info);
                    }
                    if (GST_VIDEO_INFO_FPS_N(&info) > 0 && GST_VIDEO_INFO_FPS_D(&info) > 0)
                        fps_ = static_cast<double>(GST_VIDEO_INFO_FPS_N(&info)) /
                               GST_VIDEO_INFO_FPS_D(&info);
                }
                gst_caps_unref(caps);
            }
            gst_object_unref(pad);
            if (w_ > 0 && h_ > 0) return true;
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - begin)
                           .count();
        if (elapsed > timeout_ms) return false;
        g_usleep(10000);
    }
}

bool GstDecoder::read_one_frame(VideoFrame* out, std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_ || !pipeline_ || !appsink_ || !out) {
        set_err(err, "not-opened");
        return false;
    }
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), GST_SECOND);
    if (!sample) {
        // 本地文件解码结束：appsink is-eos 为真；否则为瞬态拉取失败（可重连）
        bool eos = gst_app_sink_is_eos(GST_APP_SINK(appsink_));
        set_err(err, eos ? "eof" : "pull-sample-fail");
        return false;
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, gst_sample_get_caps(sample))) {
        set_err(err, "no-caps");
        gst_sample_unref(sample);
        return false;
    }
    w_ = GST_VIDEO_INFO_WIDTH(&info);
    h_ = GST_VIDEO_INFO_HEIGHT(&info);
    if (GST_VIDEO_INFO_FPS_N(&info) > 0 && GST_VIDEO_INFO_FPS_D(&info) > 0)
        fps_ = static_cast<double>(GST_VIDEO_INFO_FPS_N(&info)) / GST_VIDEO_INFO_FPS_D(&info);

#ifdef HAVE_NVBUF
    // L4T 硬件解码（nvv4l2decoder→nvvidconv→主机 NV12）复用下方软解 read 路径（device_only_active_=false）。
#endif

    if (device_only_active_) {
#ifdef HAVE_GSTCUDA
        // 设备直通：以 GST_MAP_CUDA 把 CUDA memory 映射为设备平面（不回主机）。
        // frame.data[0]=Y / frame.data[1]=UV 为 CUDA 设备指针，stride 为各平面步长。
        GstVideoFrame frame;
        if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ_CUDA)) {
            set_err(err, "nvcodec-map-fail");
            gst_sample_unref(sample);
            return false;
        }
        if (!frame.data[0] || !frame.data[1]) {
            set_err(err, "nvcodec-no-plane");
            gst_video_frame_unmap(&frame);
            gst_sample_unref(sample);
            return false;
        }
        std::shared_ptr<void> owner(
            frame.data[0],
            [frame, sample](void*) mutable {
                gst_video_frame_unmap(&frame);
                gst_sample_unref(sample);
            });
        IPlaneView v{static_cast<const uint8_t*>(frame.data[0]), frame.info.stride[0],
                     static_cast<const uint8_t*>(frame.data[1]), frame.info.stride[1],
                     w_, h_, Device::GPU, owner};
        out->image = make_image_from_planes_view(v);
        out->pts_ms = (GST_BUFFER_PTS_IS_VALID(buffer))
                          ? static_cast<uint64_t>(GST_BUFFER_PTS(buffer) / GST_MSECOND)
                          : 0;
        stats_.frames_out++;
        return true;
#else
        set_err(err, "nvcodec-not-supported");
        gst_sample_unref(sample);
        return false;
#endif
    }
    GstVideoFrame frame;
    if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
        set_err(err, "map-fail");
        gst_sample_unref(sample);
        return false;
    }
    // owner 在 ImageData 生命周期内保活映射的缓冲；最后一次引用释放时 unmap + 释放 sample。
    std::shared_ptr<void> owner(
        frame.data[0],
        [frame, sample](void*) mutable {
            gst_video_frame_unmap(&frame);
            gst_sample_unref(sample);
        });
    IPlaneView v{static_cast<const uint8_t*>(frame.data[0]), frame.info.stride[0],
                 static_cast<const uint8_t*>(frame.data[1]), frame.info.stride[1],
                 w_, h_, Device::CPU, owner};
    out->image = make_image_from_planes_view(v);
    out->pts_ms = (GST_BUFFER_PTS_IS_VALID(buffer))
                      ? static_cast<uint64_t>(GST_BUFFER_PTS(buffer) / GST_MSECOND)
                      : 0;
    stats_.frames_out++;
    return true;
}

void GstDecoder::set_callback(FrameCallback cb) {
    std::lock_guard<std::mutex> lk(mtx_);
    // Phase1 后端内最小实现：记录回调（异步推送留给 Phase2 线程化运行时）
    (void)cb;
}

bool GstDecoder::start(std::string* err) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!opened_) {
        set_err(err, "not-opened");
        return false;
    }
    state_ = State::Running;
    return true;
}

void GstDecoder::stop() {
    std::lock_guard<std::mutex> lk(mtx_);
    state_ = State::Idle;
}

void GstDecoder::set_device_only(bool v) {
    std::lock_guard<std::mutex> lk(mtx_);
    cfg_.device_only = v;  // 设备直通留 Phase2
}

int GstDecoder::fps() const { return static_cast<int>(fps_); }

int GstDecoder::width() const { return w_; }

int GstDecoder::height() const { return h_; }

VideoStats& GstDecoder::stats() { return stats_; }

std::string GstDecoder::last_error() const { return err_; }

void GstDecoder::close() {
    std::lock_guard<std::mutex> lk(mtx_);
    cleanup();
    close_pipeline();
    state_ = State::Closed;
}

void GstDecoder::cleanup() {
    w_ = 0;
    h_ = 0;
    fps_ = 0.0;
    opened_ = false;
    device_only_active_ = false;
#ifdef HAVE_NVBUF
    l4t_hw_active_ = false;
#endif
#ifdef ENABLE_VAAPI
    vaapi_active_ = false;
#endif
}

void GstDecoder::close_pipeline() {
    if (appsink_) { gst_object_unref(appsink_); appsink_ = nullptr; }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

void GstDecoder::set_err(std::string* err, const std::string& msg) {
    err_ = msg;
    if (err) *err = msg;
    stats_.error_count++;
}

#endif // ENABLE_GSTREAMER
} // namespace modeldeploy::video
