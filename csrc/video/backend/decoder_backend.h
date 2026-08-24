#pragma once
#include "csrc/video/video_frame.h"
#include "csrc/video/video_common.h"
#include <functional>
#include <string>

namespace modeldeploy::video {
// 解码一帧后的回调（帧以移动语义交付，消费者在回调后不再保留）
using FrameCallback = std::function<void(VideoFrame&&)>;

// 解码后端抽象接口（不泄漏 FFmpeg/GStreamer 原生结构）
class DecoderBackend {
public:
    virtual ~DecoderBackend() = default;
    virtual bool runtime_available() const = 0;
    virtual bool open(const std::string& url, std::string* err) = 0;
    virtual bool read_one_frame(VideoFrame* out, std::string* err) = 0;
    virtual void set_callback(FrameCallback cb) = 0;
    virtual bool start(std::string* err) = 0;
    virtual void stop() = 0;
    virtual void set_device_only(bool v) = 0;
    virtual int fps() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual VideoStats& stats() = 0;
    virtual void close() = 0;
    virtual std::string last_error() const = 0;
};
} // namespace modeldeploy::video
