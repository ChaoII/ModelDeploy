#include "video_sink.hpp"
#include "video_codec.hpp"
#include "csrc/video/video_frame.h"
using namespace modeldeploy;

VideoSink::~VideoSink() { close(); }

bool VideoSink::open(const std::string& url, int w, int h, int src_fps,
                     const EncoderConfig& ec, bool gpu_direct, std::string* err) {
    modeldeploy::video::VideoEncoderConfig sdk;
    video_codec_fill_encoder(&sdk, ec, gpu_direct);
    enc_ = modeldeploy::video::VideoEncoder::create(sdk, err);
    if (!enc_) return false;
    if (!enc_->open(url, w, h, src_fps, err)) { enc_.reset(); return false; }
    return true;
}

bool VideoSink::encode(const modeldeploy::vision::ImageData& image, std::string* err) {
    if (!enc_) { if (err) *err = "sink-not-open"; return false; }
    return enc_->encode_async(image);
}

bool VideoSink::start_async(std::string* err) { return enc_ ? enc_->start_async(err) : false; }
void VideoSink::stop_async() { if (enc_) enc_->stop_async(); }
void VideoSink::close() { if (enc_) enc_->close(); }
modeldeploy::video::State VideoSink::state() const { return enc_ ? enc_->state() : modeldeploy::video::State::Closed; }
std::string VideoSink::last_error() const { return enc_ ? enc_->last_error() : ""; }
const modeldeploy::video::VideoStats& VideoSink::stats() const {
    static const modeldeploy::video::VideoStats kEmpty{};
    return enc_ ? enc_->stats() : kEmpty;
}
bool VideoSink::has_failed() const { return enc_ ? enc_->has_permanently_failed() : true; }
