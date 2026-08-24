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
    return std::shared_ptr<VideoDecoder>(new VideoDecoder(std::move(backend)));
}

VideoDecoder::VideoDecoder(std::shared_ptr<DecoderBackend> b)
    : backend_(std::move(b)) {}

VideoDecoder::~VideoDecoder() { close(); }

bool VideoDecoder::open(const std::string& url, std::string* err) {
    state_ = State::Opening;
    bool ok = backend_ ? backend_->open(url, err) : false;
    if (!ok && err && err->empty()) *err = "not-initialized";
    state_ = ok ? State::Running : State::Error;
    return ok;
}

bool VideoDecoder::read_one_frame(VideoFrame* out, std::string* err) {
    if (!backend_) return false;
    bool ok = backend_->read_one_frame(out, err);
    if (!ok) state_ = State::Eof;
    return ok;
}

void VideoDecoder::set_callback(FrameCallback cb) {
    if (backend_) backend_->set_callback(std::move(cb));
}

bool VideoDecoder::start(std::string* err) {
    if (!backend_) return false;
    bool ok = backend_->start(err);
    state_ = ok ? State::Running : State::Error;
    return ok;
}

void VideoDecoder::stop() {
    if (backend_) backend_->stop();
    state_ = State::Idle;
}

void VideoDecoder::set_device_only(bool v) {
    if (backend_) backend_->set_device_only(v);
}

State VideoDecoder::state() const { return state_; }

const VideoStats& VideoDecoder::stats() const { return backend_->stats(); }

int VideoDecoder::fps() const { return backend_ ? backend_->fps() : 0; }

int VideoDecoder::width() const { return backend_ ? backend_->width() : 0; }

int VideoDecoder::height() const { return backend_ ? backend_->height() : 0; }

void VideoDecoder::close() {
    if (backend_) {
        backend_->close();
        state_ = State::Closed;
    } else {
        state_ = State::Idle;
    }
}

} // namespace modeldeploy::video
