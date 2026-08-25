#pragma once
#include "csrc/video/video_common.h"
#include "csrc/video/video_frame.h"
#include "vision/common/image_data.h"
#include <cstdint>
#include <string>

namespace modeldeploy::video {
// 编码后端抽象接口（不泄漏 FFmpeg/GStreamer 原生结构）
class EncoderBackend {
public:
    virtual ~EncoderBackend() = default;
    virtual bool runtime_available() const = 0;
    virtual bool open(const std::string& output_url, int w, int h, int src_fps,
                      const VideoEncoderConfig& cfg, std::string* err) = 0;
    // GPU（device==Device::GPU）输入的 ImageData 平面借用自调用方设备内存，SDK 以
    // 包装/直通方式编码而不接管其生命周期；调用方必须保证这些平面在 close() 之前有效。
    // CPU 输入由 SDK 拷贝处理，无此约束。
    virtual bool encode(const VideoFrame& frame, std::string* err) = 0;
    virtual bool encode_async(const modeldeploy::vision::ImageData& image) = 0;
    virtual bool start_async(std::string* err) = 0;
    virtual void stop_async() = 0;
    virtual bool has_permanently_failed() const = 0;
    virtual VideoStats& stats() = 0;
    virtual std::string last_error() const = 0;
    virtual void close() = 0;
};
} // namespace modeldeploy::video
