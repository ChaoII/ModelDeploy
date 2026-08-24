#pragma once
#include "csrc/video/video_codec_config.h"
#include "csrc/video/backend/encoder_backend.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

// GStreamer C 头仅在本后端内使用，绝不泄漏给门面/使用者侧。
// ENABLE_GSTREAMER=OFF 时本类不引用任何 GStreamer 符号（gst_encoder.cpp 实现整体为空）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#endif

namespace modeldeploy::video {

// GStreamer 软编后端（x264enc）：CPU BGR → appsrc → videoconvert → x264enc → mp4mux → filesink。
// 用 gst_app_src_push_buffer 阻塞推帧形成背压；close 时刷 EOS 并等 mp4mux 收尾写出 moov 再停管道。
// 本类经 factory 在 DLL 内以 make_shared 实例化，返回 shared_ptr<EncoderBackend> 给外部，
// 无需导出（与 EncoderBackend 接口一致，均不导出）。
class GstEncoder : public EncoderBackend {
public:
    explicit GstEncoder(const VideoEncoderConfig& cfg);
    ~GstEncoder() override;

    bool runtime_available() const override;
    bool open(const std::string& url, int w, int h, int src_fps,
              const VideoEncoderConfig& c, std::string* err) override;
    bool encode(const modeldeploy::vision::ImageData& image, std::string* err) override;
    bool encode_from_gpu_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                              std::string* err) override;
    bool encode_async(const modeldeploy::vision::ImageData& image) override;
    bool start_async(std::string* err) override;
    void stop_async() override;
    bool has_permanently_failed() const override;
    VideoStats& stats() override;
    std::string last_error() const override;
    void close() override;

    // 静态探测：gst_init 一次 + 检查编码所需插件（appsrc/x264enc/mp4mux 等）可实例化
    static bool gstreamer_x264_available();

private:
    void build_pipeline(const std::string& url, int w, int h, int fps);
    bool start_pipeline(std::string* err);
    void wait_eos_and_stop();  // 刷 EOS 后等待 EOS/错误消息（带超时）并停管道
    void teardown();           // 停管道并释放 pipeline/appsrc/bus
    void set_err(std::string* err, const std::string& msg);

    VideoEncoderConfig cfg_;
    int w_ = 0, h_ = 0, fps_ = 0;
    uint64_t pts_ = 0;
    VideoStats stats_;
    std::string err_;
    std::mutex mtx_;
    std::atomic<bool> opened_{false};

#ifdef ENABLE_GSTREAMER
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstBus* bus_ = nullptr;
#endif
};

} // namespace modeldeploy::video
