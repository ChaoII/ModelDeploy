#include "csrc/video/video_codec_config.h"

namespace modeldeploy::video {

namespace {
const char* const kKnownFormats[] = {"auto", "rtsp", "rtmp", "flv", "mp4"};
bool known_format(const std::string& f) {
    for (const char* fmt : kKnownFormats) {
        if (f == fmt) return true;
    }
    return false;
}
} // namespace

bool VideoDecoderConfig::validate(std::string* err) const {
    if (reconnect_delay_ms < 0) {
        if (err) *err = "reconnect_delay_ms must be >= 0";
        return false;
    }
    if (max_reconnects < 0) {
        if (err) *err = "max_reconnects must be >= 0";
        return false;
    }
    if (timeout_us <= 0) {
        if (err) *err = "timeout_us must be > 0";
        return false;
    }
    return true;
}

bool VideoEncoderConfig::validate(std::string* err) const {
    if (fps < 0) {
        if (err) *err = "fps must be >= 0";
        return false;
    }
    if (fps == 0) {
        if (err) *err = "fps must be set (>0)";
        return false;
    }
    if (bitrate_kbps <= 0) {
        if (err) *err = "bitrate_kbps must be > 0";
        return false;
    }
    if (!known_format(format)) {
        if (err) *err = "unknown format: " + format;
        return false;
    }
    return true;
}

} // namespace modeldeploy::video
