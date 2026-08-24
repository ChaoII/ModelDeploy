#include "csrc/video/backend/gst_decoder.h"

// ENABLE_GSTREAMER=OFF 时整个翻译单元编译为空（不引用任何 GStreamer 符号）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <chrono>
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
        // 本地文件解码结束：appsink sink pad 的 last-flow-return 为 EOS
        bool eos = false;
        GstPad* sinkpad = gst_element_get_static_pad(appsink_, "sink");
        if (sinkpad) {
            eos = gst_pad_get_last_flow_return(sinkpad) == GST_FLOW_EOS;
            gst_object_unref(sinkpad);
        }
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
