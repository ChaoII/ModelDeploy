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

// GPU 直编（CUDA memory）用：仅声明 Opaque 指针，避免把 gst/cuda 头暴露给本头（其夹带 cudaGL.h 与 GL/gl.h 冲突）。
// HAVE_GSTCUDA 由 cmake 在探测到 gstcuda 时定义；未定义时本类不携带任何 CUDA 状态。
#ifdef HAVE_GSTCUDA
struct _GstCudaContext;
struct _GstCudaAllocator;
#endif

namespace modeldeploy::video {

// GStreamer 编码后端：CPU BGR → appsrc → videoconvert → (x264enc|nvh264enc) → h264parse → mp4mux → filesink。
// open() 依据 hw_accel∈{Auto,Cuda} 且 nvh264enc 可实例化（或显式 codec=nvh264enc）决议用 nv 硬编，
// 否则回退软编 x264enc。用 gst_app_src_push_buffer 阻塞推帧形成背压；close 时刷 EOS 并等
// mp4mux 收尾写出 moov 再停管道。本类经 factory 在 DLL 内以 make_shared 实例化，返回
// shared_ptr<EncoderBackend> 给外部，无需导出（与 EncoderBackend 接口一致，均不导出）。
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

    // 本次会话是否实际启用了硬件（nvh264enc / vaapih264enc）编码器；false 表示走了软编（x264enc）
    bool used_hw() const {
#ifdef ENABLE_VAAPI
        return encoder_is_nv_ || encoder_is_vaapi_;
#else
        return encoder_is_nv_;
#endif
    }

    // 静态探测：gst_init 一次 + 检查编码所需插件（appsrc/x264enc/mp4mux 等）可实例化
    static bool gstreamer_x264_available();
    // 静态探测：nvcodec 硬编插件 nvh264enc 可实例化（复用 H0 同款 gst_element_factory_find 检查）
    static bool nvh264enc_available();
#ifdef ENABLE_VAAPI
    // 静态探测：vaapih264enc 插件可实例化（未在本机验证，需 Linux GStreamer vaapi 插件）
    static bool vaapih264enc_available();
#endif

private:
    // 决议本次会话用的编码元素：0=软编 x264enc，1=硬编 nvh264enc，2=VAAPI vaapih264enc，-1=错误（err 已设置）
    int resolve_encoder(std::string* err);
    void build_pipeline(const std::string& url, int w, int h, int fps, int enc);
    bool start_pipeline(std::string* err);
    // GPU 直编（gpu_direct_input）专用：以 CUDA memory 包装设备 NV12 指针并推入 appsrc。
    // 返回 false 且 err 已设置。
    bool encode_from_gpu_nv12_impl(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                                   std::string* err);
    void wait_eos_and_stop();  // 刷 EOS 后等待 EOS/错误消息（带超时）并停管道
    void teardown();           // 停管道并释放 pipeline/appsrc/bus
    void set_err(std::string* err, const std::string& msg);

    VideoEncoderConfig cfg_;
    int w_ = 0, h_ = 0, fps_ = 0;
    uint64_t pts_ = 0;
    bool encoder_is_nv_ = false;      // 本次会话是否实际用了 nvh264enc 硬编
#ifdef ENABLE_VAAPI
    bool encoder_is_vaapi_ = false;   // 本次会话是否实际用了 vaapih264enc 硬编
#endif
    VideoStats stats_;
    std::string err_;
    std::mutex mtx_;
    std::atomic<bool> opened_{false};

#ifdef ENABLE_GSTREAMER
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstBus* bus_ = nullptr;
#endif
#ifdef HAVE_GSTCUDA
    // 会话内 CUDA 上下文/分配器（GPU 直编专用）。作为成员而非进程级 static：
    // 避免在 DLL/测试进程生命期里全局持有，防止污染同进程后建的 nvh264enc 管道。
    _GstCudaContext* cuda_ctx_ = nullptr;
    _GstCudaAllocator* cuda_alloc_ = nullptr;
#endif
};

} // namespace modeldeploy::video
