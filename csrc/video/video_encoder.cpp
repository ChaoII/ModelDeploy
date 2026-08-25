#include "csrc/video/video_encoder.h"
#include "csrc/video/backend/encoder_backend.h"
#include "csrc/video/factory.h"
#include <utility>

namespace modeldeploy::video {

std::shared_ptr<VideoEncoder> VideoEncoder::create(const VideoEncoderConfig& cfg,
                                                   std::string* err) {
    auto backend = create_encoder_backend(cfg);
    if (!backend) {
        if (err) *err = "backend unavailable";
        return nullptr;
    }
    return std::shared_ptr<VideoEncoder>(new VideoEncoder(std::move(backend), cfg));
}

VideoEncoder::VideoEncoder(std::shared_ptr<EncoderBackend> b, VideoEncoderConfig cfg)
    : cfg_(std::move(cfg)), backend_(std::move(b)) {}

VideoEncoder::~VideoEncoder() { close(); }

bool VideoEncoder::open(const std::string& url, int w, int h, int src_fps, std::string* err) {
    state_ = State::Opening;
    bool ok = backend_ ? backend_->open(url, w, h, src_fps, cfg_, err) : false;
    if (!ok && err && err->empty()) *err = "not-initialized";
    state_ = ok ? State::Running : State::Error;
    return ok;
}

bool VideoEncoder::encode(const VideoFrame& frame, std::string* err) {
    if (!backend_) return false;
    bool ok = backend_->encode(frame, err);
    if (!ok) state_ = State::Error;
    return ok;
}

bool VideoEncoder::encode_async(const modeldeploy::vision::ImageData& image) {
    return backend_ ? backend_->encode_async(image) : false;
}

bool VideoEncoder::start_async(std::string* err) {
    if (!backend_) return false;
    bool ok = backend_->start_async(err);
    state_ = ok ? State::Running : State::Error;
    return ok;
}

void VideoEncoder::stop_async() {
    if (backend_) backend_->stop_async();
    state_ = State::Idle;
}

bool VideoEncoder::has_permanently_failed() const {
    return backend_ ? backend_->has_permanently_failed() : false;
}

State VideoEncoder::state() const { return state_; }

std::string VideoEncoder::last_error() const { return backend_ ? backend_->last_error() : ""; }

const VideoStats& VideoEncoder::stats() const { return backend_->stats(); }

void VideoEncoder::close() {
    if (backend_) {
        backend_->close();
        state_ = State::Closed;
    } else {
        state_ = State::Idle;
    }
}

} // namespace modeldeploy::video
