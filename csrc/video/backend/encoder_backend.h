#pragma once
#include "csrc/video/video_common.h"
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
    virtual bool encode(const modeldeploy::vision::ImageData& image, std::string* err) = 0;
    virtual bool encode_from_gpu_nv12(const uint8_t* d_y, const uint8_t* d_uv, int w, int h,
                                      std::string* err) = 0;
    virtual bool encode_async(const modeldeploy::vision::ImageData& image) = 0;
    virtual bool start_async(std::string* err) = 0;
    virtual void stop_async() = 0;
    virtual bool has_permanently_failed() const = 0;
    virtual VideoStats& stats() = 0;
    virtual std::string last_error() const = 0;
    virtual void close() = 0;
};
} // namespace modeldeploy::video
