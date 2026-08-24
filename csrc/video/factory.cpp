#include "csrc/video/factory.h"
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/backend/encoder_backend.h"
#include "csrc/video/video_common.h"

namespace modeldeploy::video {

VideoCodecCapabilities query_video_capabilities() {
    VideoCodecCapabilities cap;
    // backend 探测在 Task3-6 填充 runtime_available 后接入
    return cap;
}

std::shared_ptr<DecoderBackend> create_decoder_backend(const VideoDecoderConfig& cfg) {
    if (cfg.backend == CodecBackend::GStreamer) return nullptr;  // Task5 接入
    return nullptr;                                              // Task3 接入
}

std::shared_ptr<EncoderBackend> create_encoder_backend(const VideoEncoderConfig& cfg) {
    if (cfg.backend == CodecBackend::GStreamer) return nullptr;  // Task6 接入
    return nullptr;                                              // Task4 接入
}

} // namespace modeldeploy::video
