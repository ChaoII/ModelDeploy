#include "csrc/video/video_decoder.h"
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/factory.h"
#include <utility>

namespace modeldeploy::video {

std::shared_ptr<VideoDecoder> VideoDecoder::create(const VideoDecoderConfig& cfg,
                                                   std::string* err) {
    auto backend = create_decoder_backend(cfg);
    if (!backend) {
        if (err) *err = "backend unavailable";
        return nullptr;
    }
    return std::shared_ptr<VideoDecoder>(new VideoDecoder(std::move(backend), cfg));
}

VideoDecoder::VideoDecoder(std::shared_ptr<DecoderBackend> b, const VideoDecoderConfig& cfg)
    : pipeline_(std::move(b), cfg) {}

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open(const std::string& url, std::string* err) {
    return pipeline_.open(url, err);
}

bool VideoDecoder::read_one_frame(VideoFrame* out, std::string* err) {
    return pipeline_.read_one_frame(out, err);
}

void VideoDecoder::set_callback(FrameCallback cb) { pipeline_.set_callback(std::move(cb)); }

bool VideoDecoder::start(std::string* err) { return pipeline_.start(err); }

void VideoDecoder::stop() { pipeline_.stop(); }

void VideoDecoder::set_device_only(bool v) { pipeline_.set_device_only(v); }

State VideoDecoder::state() const { return pipeline_.state(); }

const VideoStats& VideoDecoder::stats() const { return pipeline_.stats(); }

std::string VideoDecoder::last_error() const { return pipeline_.last_error(); }

int VideoDecoder::fps() const { return pipeline_.fps(); }

int VideoDecoder::width() const { return pipeline_.width(); }

int VideoDecoder::height() const { return pipeline_.height(); }

void VideoDecoder::close() { pipeline_.close(); }

uint64_t VideoDecoder::pool_hits() const { return pipeline_.pool_hits(); }

uint64_t VideoDecoder::pool_returns() const { return pipeline_.pool_returns(); }

} // namespace modeldeploy::video
