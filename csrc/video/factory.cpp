#include "csrc/video/factory.h"
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/backend/encoder_backend.h"
#include "csrc/video/backend/ffmpeg_decoder.h"
#include "csrc/video/backend/ffmpeg_encoder.h"
#include "csrc/video/video_common.h"
#ifdef ENABLE_GSTREAMER
#include "csrc/video/backend/gst_decoder.h"
#include "csrc/video/backend/gst_encoder.h"
#endif

namespace modeldeploy::video {

VideoCodecCapabilities query_video_capabilities() {
    VideoCodecCapabilities cap;
    // FFmpeg 随 BUILD_VIDEO 编译；GStreamer 运行时可用性由插件探测决定
    cap.ffmpeg_available = true;
#ifdef ENABLE_GSTREAMER
    cap.gstreamer_available = GstDecoder::gstreamer_available();
#endif
    return cap;
}

std::shared_ptr<DecoderBackend> create_decoder_backend(const VideoDecoderConfig& cfg) {
#ifdef ENABLE_GSTREAMER
    if (cfg.backend == CodecBackend::GStreamer) return std::make_shared<GstDecoder>(cfg);
#endif
    return std::make_shared<FfmpegDecoder>(cfg);
}

std::shared_ptr<EncoderBackend> create_encoder_backend(const VideoEncoderConfig& cfg) {
#ifdef ENABLE_GSTREAMER
    if (cfg.backend == CodecBackend::GStreamer) return std::make_shared<GstEncoder>(cfg);
#endif
    return std::make_shared<FfmpegEncoder>(cfg);                 // Task4 FFmpeg 软编
}

} // namespace modeldeploy::video
