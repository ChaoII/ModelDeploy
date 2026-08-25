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
    void cleanup();
    // 依据 codec_id 映射 CUVID 硬解名（H264/HEVC/AV1）；其它编解码器返回空串（无硬解名）
    std::string hw_decoder_name(int codec_id) const;
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
    std::mutex mtx_;
    std::atomic<bool> opened_{false};
    bool used_hw_ = false;  // 本次会话是否实际起到硬件（CUVID）解码
};

} // namespace modeldeploy::video
