#pragma once
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/adapter.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

// GStreamer C 头仅在本后端内使用，绝不泄漏给门面/使用者侧。
// ENABLE_GSTREAMER=OFF 时本类不引用任何 GStreamer 符号（gst_decoder.cpp 实现整体为空）。
#ifdef ENABLE_GSTREAMER
#include <gst/gst.h>
#endif

namespace modeldeploy::video {

// GStreamer 软解后端：文件 → decodebin → videoconvert → NV12 → VideoFrame
// （appsink pull-sample + gst_video_frame_map 零拷贝进 ImageData，owner 保活 GstSample）。
// 本类经 factory 在 DLL 内以 make_shared 实例化，返回 shared_ptr<DecoderBackend> 给外部，
// 无需导出（与 DecoderBackend 接口一致，均不导出）。
class GstDecoder : public DecoderBackend {
public:
    explicit GstDecoder(const VideoDecoderConfig& cfg);
    ~GstDecoder() override;

    bool runtime_available() const override;
    bool open(const std::string& url, std::string* err) override;
    bool read_one_frame(VideoFrame* out, std::string* err) override;
    void set_callback(FrameCallback cb) override;
    bool start(std::string* err) override;
    void stop() override;
    void set_device_only(bool v) override;
    int fps() const override;
    int width() const override;
    int height() const override;
    VideoStats& stats() override;
    std::string last_error() const override;
    void close() override;

    // 静态探测：进程内 gst_init 一次 + 检查所需插件是否可实例化
    static bool gstreamer_available();

private:
    void cleanup();                 // 重置尺寸/状态（不释放管道）
    void close_pipeline();          // 停管道并释放 pipeline/appsink
    bool query_caps_locked(int timeout_ms);  // 从 appsink sink pad 的 current caps 取宽高/帧率
#ifdef HAVE_GSTCUDA
    // 设备直通：构建 nvh264dec → CUDA memory → appsink(memory:CUDAMemory) 管道；成功置 device_only_active_
    bool build_device_pipeline_locked(const std::string& url, std::string* err);
#endif
#ifdef HAVE_NVBUF
    // Jetson L4T：nvv4l2decoder 硬件解码 + nvvidconv → 主机 NV12（GStreamer 标准取帧，L4T 无 CUDA 零拷贝，
    // 故转主机帧；decode 仍硬件加速）。复用软解 read 路径（device_only_active_ 保持 false）。
    static bool nvv4l2decoder_available();
    bool build_hwdecode_pipeline_locked(const std::string& url, std::string* err);
#endif
#ifdef ENABLE_VAAPI
    // 静态探测：vaapih264dec 插件可实例化（未在本机验证，需 Linux GStreamer vaapi 插件）
    static bool vaapih264dec_available();
    // VAAPI 硬解：filesrc → h264parse → vaapih264dec → videoconvert → appsink(NV12)，输出 CPU NV12。
    // 复用软解 read 路径（device_only_active_ 保持 false）。未在本机验证（需 Linux VAAPI）。
    bool build_vaapi_pipeline_locked(const std::string& url, std::string* err);
#endif
    void set_err(std::string* err, const std::string& msg);

#ifdef ENABLE_GSTREAMER
    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;
#endif
    VideoDecoderConfig cfg_;
    State state_ = State::Idle;
    int w_ = 0, h_ = 0;
    double fps_ = 0.0;
    VideoStats stats_;
    std::string err_;
    std::mutex mtx_;
    std::atomic<bool> opened_{false};
    bool device_only_active_ = false;  // 设备直通模式：输出保持 CUDA 设备帧
#ifdef HAVE_NVBUF
    bool l4t_hw_active_ = false;   // Jetson L4T 硬件解码（nvv4l2decoder → nvvidconv → 主机 NV12）
#endif
#ifdef ENABLE_VAAPI
    bool vaapi_active_ = false;  // 本次会话是否实际用 VAAPI 硬解
#endif
};

} // namespace modeldeploy::video
