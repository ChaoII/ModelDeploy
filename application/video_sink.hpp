#pragma once
#include <memory>
#include <string>
#include "config.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/video/video_encoder.h"

class VideoSink {
public:
    VideoSink() = default;
    ~VideoSink();

    bool open(const std::string& url, int w, int h, int src_fps,
              const EncoderConfig& ec, bool gpu_direct, std::string* err = nullptr);
    bool encode(const modeldeploy::vision::ImageData& image, std::string* err = nullptr);

    bool start_async(std::string* err = nullptr);
    void stop_async();
    void close();

    modeldeploy::video::State state() const;
    std::string last_error() const;
    const modeldeploy::video::VideoStats& stats() const;
    bool has_failed() const;

private:
    std::shared_ptr<modeldeploy::video::VideoEncoder> enc_;
};
