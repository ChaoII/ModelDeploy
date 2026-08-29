#pragma once
#include <functional>
#include <memory>
#include <string>
#include "config.hpp"
#include "csrc/video/video_decoder.h"
#include "csrc/video/video_frame.h"

class VideoSource {
public:
    using FrameCallback = std::function<void(modeldeploy::video::VideoFrame&&)>;
    VideoSource() = default;
    ~VideoSource();

    bool open(const std::string& url, const DecoderConfig& dc, std::string* err = nullptr);
    void set_callback(FrameCallback cb);
    bool start(std::string* err = nullptr);
    void stop();
    void close();

    modeldeploy::video::State state() const;
    std::string last_error() const;
    const modeldeploy::video::VideoStats& stats() const;
    int fps() const;
    int width() const;
    int height() const;

private:
    std::shared_ptr<modeldeploy::video::VideoDecoder> dec_;
};
