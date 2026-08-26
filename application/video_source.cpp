#include "video_source.hpp"
#include "video_codec.hpp"
using namespace modeldeploy;

VideoSource::~VideoSource() { close(); }

bool VideoSource::open(const std::string& url, const DecoderConfig& dc, std::string* err) {
    modeldeploy::video::VideoDecoderConfig sdk;
    video_codec_fill_decoder(&sdk, dc);
    dec_ = modeldeploy::video::VideoDecoder::create(sdk, err);
    if (!dec_) return false;
    if (!dec_->open(url, err)) { dec_.reset(); return false; }
    return true;
}

void VideoSource::set_callback(FrameCallback cb) {
    if (dec_) dec_->set_callback(std::move(cb));
}

bool VideoSource::start(std::string* err) { return dec_ ? dec_->start(err) : false; }
void VideoSource::stop() { if (dec_) dec_->stop(); }
void VideoSource::close() { if (dec_) dec_->close(); }

modeldeploy::video::State VideoSource::state() const {
    return dec_ ? dec_->state() : modeldeploy::video::State::Closed;
}
std::string VideoSource::last_error() const { return dec_ ? dec_->last_error() : ""; }
int VideoSource::fps() const { return dec_ ? dec_->fps() : 0; }
const modeldeploy::video::VideoStats& VideoSource::stats() const {
    static const modeldeploy::video::VideoStats kEmpty{};
    return dec_ ? dec_->stats() : kEmpty;
}
