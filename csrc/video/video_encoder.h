#pragma once

#include "csrc/video/video_common.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_frame.h"
#include "vision/common/image_data.h"
#include <memory>
#include <string>

namespace modeldeploy::video {

class EncoderBackend;

// SDK 视频编码门面（RAII 包装 EncoderBackend）：后端无关，接口不泄漏 FFmpeg/GStreamer 原生结构。
class MODELDEPLOY_CXX_EXPORT VideoEncoder {
public:
    // 按配置创建后端并返回门面；后端不可用时 set err 并返回 nullptr
    static std::shared_ptr<VideoEncoder> create(const VideoEncoderConfig& cfg,
                                                std::string* err = nullptr);
    ~VideoEncoder();

    bool open(const std::string& url, int w, int h, int src_fps, std::string* err = nullptr);
    // 编码一帧（CPU BGR/设备 NV12 + 可选 pts_ms）到输出容器；失败返回 false。
    // GPU（device==Device::GPU）输入：SDK 借用调用方设备平面直通编码、不持有其生命周期，
    // 调用方必须保证这些平面在 close() 之前有效。CPU 输入由 SDK 拷贝，无此约束。
    bool encode(const VideoFrame& frame, std::string* err = nullptr);
    bool encode_async(const modeldeploy::vision::ImageData& image);
    bool start_async(std::string* err = nullptr);
    void stop_async();
    bool has_permanently_failed() const;

    State state() const;
    std::string last_error() const;
    const VideoStats& stats() const;
    void close();  // 幂等

private:
    explicit VideoEncoder(std::shared_ptr<EncoderBackend> b, VideoEncoderConfig cfg);
    VideoEncoderConfig cfg_;
    std::shared_ptr<EncoderBackend> backend_;
    mutable State state_ = State::Idle;
};

} // namespace modeldeploy::video
