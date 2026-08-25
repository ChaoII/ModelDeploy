#pragma once
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/adapter.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>

// FFmpeg C 头仅在本后端内使用，绝不泄漏给门面/使用者侧。
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
}

namespace modeldeploy::video {

// FFmpeg 软解后端：h264/h265 → NV12 → VideoFrame（零拷贝进 ImageData，owner 保活解码缓冲）。
// 本类经 factory 在 DLL 内以 make_shared 实例化，返回 shared_ptr<DecoderBackend> 给外部，
// 无需导出（与 DecoderBackend 接口一致，均不导出）。
class FfmpegDecoder : public DecoderBackend {
public:
    explicit FfmpegDecoder(const VideoDecoderConfig& cfg);
    ~FfmpegDecoder() override;

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

    // 本次会话是否实际使用了硬件（CUVID 名）解码器；false 表示走了软解（含硬解不可用的降级）
    bool used_hw() const { return used_hw_; }

private:
    // 打开核心逻辑：假设调用方已持有 mtx_（open()/close() 加锁后调用；运行期回退在锁内直呼）。
    // reset_fallback==true（外部全新 open）时复位 delivered_frames_ 与 hw_fallback_done_；
    // 内部运行期回退重开传 false，保留"只回退一次"语义。
    bool open_locked(const std::string& url, std::string* err, bool reset_fallback);
    // 读一帧核心逻辑：假设调用方已持有 mtx_（read_one_frame 加锁后调用）。回退重开后自行重入。
    bool read_one_frame_locked(VideoFrame* out, std::string* err);
    void cleanup();
    // 依据 codec_id 映射 CUVID 硬解名（H264/HEVC/AV1）；其它编解码器返回空串（无硬解名）
    std::string hw_decoder_name(int codec_id) const;
    // 设备直通：以 CUDA 硬件设备上下文打开 hwc（h264_cuvid 等），使解码输出 AV_PIX_FMT_CUDA 设备帧。
    // 成功返回 true 且 ctx_ 持有 hw_device_ctx 引用；否则返回 false（不留下 ctx_）。
    bool setup_cuda_device_decoder(AVCodecParameters* cp, const AVCodec* hwc);
    // 把 frame_（非 NV12）经 swscale 转成 NV12 存到 sws_frame_；成功返回 true
    bool convert_to_nv12();
    void set_err(std::string* err, const std::string& msg);

    VideoDecoderConfig cfg_;
    State state_ = State::Idle;
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* ctx_ = nullptr;
    int vstream_ = -1;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;        // 解码器输出帧（被 read 内 ref 到 owner）
    AVFrame* sws_frame_ = nullptr;    // swscale 转换目标（NV12），持有自有 buffer
    SwsContext* sws_ = nullptr;       // swscale 上下文（源格式变化时重建）
    int sws_fmt_ = -1;                // sws_ 对应的源像素格式
    int w_ = 0, h_ = 0;
    double fps_ = 25.0;
    VideoStats stats_;
    std::string err_;
    std::string url_;       // 当前已打开的 URL（运行期 0 帧回退需重开同一地址）
    std::mutex mtx_;
    std::atomic<bool> opened_{false};
    bool used_hw_ = false;  // 本次会话是否实际起到硬件（CUVID）解码
    bool device_only_active_ = false;  // 设备直通模式：解码输出保持 GPU 设备帧（AV_PIX_FMT_CUDA）
    bool force_soft_ = false;            // 强制软解标志（运行期回退重开时置位，跳过硬件选择）
    bool hw_fallback_done_ = false;      // 运行期 0 帧回退软解是否已发生（至多一次）
    uint64_t delivered_frames_ = 0;      // 本次会话已交付的帧数（每次成功交付时同步 +1）
};

} // namespace modeldeploy::video
