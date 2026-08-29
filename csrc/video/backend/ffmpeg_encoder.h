#pragma once
#include "csrc/video/video_codec_config.h"
#include "csrc/video/backend/encoder_backend.h"

// FFmpeg C 头仅在本后端内使用，绝不泄漏给门面/使用者侧。
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
namespace modeldeploy::video {

// FFmpeg 软编后端（libx264）：CPU BGR → YUV420P 编码 → 封装输出。
// 本类经 factory 在 DLL 内以 make_shared 实例化，返回 shared_ptr<EncoderBackend> 给外部，
// 无需导出（与 EncoderBackend 接口一致，均不导出）。
class FfmpegEncoder : public EncoderBackend {
public:
    explicit FfmpegEncoder(const VideoEncoderConfig& cfg);
    ~FfmpegEncoder() override;

    bool runtime_available() const override;
    bool open(const std::string& url, int w, int h, int src_fps,
              const VideoEncoderConfig& c, std::string* err) override;
    bool encode(const VideoFrame& frame, std::string* err) override;
    bool encode_async(const modeldeploy::vision::ImageData& image) override;
    bool start_async(std::string* err) override;
    void stop_async() override;
    bool has_permanently_failed() const override;
    VideoStats& stats() override;
    std::string last_error() const override;
    void close() override;

    // 本次会话是否实际启用了硬件（NVENC）编码器；false 表示走了软编（libx264）
    bool used_hw() const { return used_hw_; }

private:
    bool init_encoder(int w, int h, int fps, std::string* err);
    // kind: 0=软编 libx264, 1=NVENC(nvenc), 2=VAAPI(h264_vaapi)
    bool configure_encoder(const std::string& name, int kind, int w, int h, int fps);
    // pts_ms：非 0=调用方注入时间戳（毫秒）；0=内部按帧序自增（CFR）
    bool encode_cpu(const modeldeploy::vision::ImageData& image, uint64_t pts_ms, std::string* err);  // 软编（CPU BGR）
    bool encode_gpu(const modeldeploy::vision::ImageData& image, uint64_t pts_ms, std::string* err);  // NVENC CUDA 直通
    bool setup_cuda_hw_frames(int w, int h);
    bool d2d_copy_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h, AVFrame* hw);
#ifdef ENABLE_VAAPI
    // 创建 VAAPI 设备上下文 + NV12 hw 帧上下文；成功置 vaapi_hw_ctx_/vaapi_hw_frames_
    bool setup_vaapi_hw_frames(int w, int h);
    // 把 CPU NV12 帧上传为 VAAPI hw 帧并送入编码器（发送/收包循环）。未在本机验证（需 Linux VAAPI）。
    bool send_vaapi_frame(std::string* err);
#endif
    bool open_output(const std::string& url);
    void cleanup();
    void set_err(std::string* err, const std::string& msg);

    VideoEncoderConfig cfg_;
    int w_ = 0, h_ = 0;
    int64_t pts_ = 0;
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* enc_ = nullptr;
    AVStream* st_ = nullptr;
    SwsContext* sws_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    // GPU 直接编码（nvenc + CUDA hw_frames_ctx）：设备 NV12 指针直编会话资源
    AVBufferRef* hw_device_ctx_ = nullptr;  // CUDA 设备上下文
    AVBufferRef* hw_frames_ctx_ = nullptr;  // CUDA hw帧上下文（format=CUDA, sw_format=NV12）
#ifdef ENABLE_VAAPI
    AVBufferRef* vaapi_hw_ctx_ = nullptr;     // VAAPI 设备上下文
    AVBufferRef* vaapi_hw_frames_ = nullptr;  // VAAPI hw帧上下文（format=VAAPI, sw_format=NV12）
    bool vaapi_active_ = false;               // 本次会话是否实际用 VAAPI 硬编
#endif
    AVPixelFormat dst_fmt_ = AV_PIX_FMT_YUV420P;  // sws 输出/编码器输入 pix_fmt（随编码器）
    bool header_ = false;
    bool opened_ = false;
    bool used_hw_ = false;  // 本次会话是否实际用到硬件（NVENC/VAAPI）编码器
    VideoStats stats_;
    double encode_avg_sum_ = 0.0;   // 编码耗时累计（供除以帧数得到真实平均）
    std::string err_;
};

} // namespace modeldeploy::video
