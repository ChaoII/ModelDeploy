#include "csrc/video/factory.h"
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/backend/encoder_backend.h"
#include "csrc/video/backend/ffmpeg_decoder.h"
#include "csrc/video/backend/ffmpeg_encoder.h"
#include "csrc/video/hw_probe.h"
#include "csrc/video/video_common.h"
#include <algorithm>
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
    // 硬件能力由 hw_probe 真实探测填充（去重拼接 FFmpeg + GStreamer 两源）
    auto f = probe_ffmpeg_hw();
    cap.hw_decoders = std::move(f.decoders);
    cap.hw_encoders = std::move(f.encoders);
    auto g = probe_gstreamer_hw();
    for (const auto& n : g.decoders)
        if (std::find(cap.hw_decoders.begin(), cap.hw_decoders.end(), n) == cap.hw_decoders.end())
            cap.hw_decoders.push_back(n);
    for (const auto& n : g.encoders)
        if (std::find(cap.hw_encoders.begin(), cap.hw_encoders.end(), n) == cap.hw_encoders.end())
            cap.hw_encoders.push_back(n);
    return cap;
}

std::shared_ptr<DecoderBackend> create_decoder_backend(const VideoDecoderConfig& cfg) {
    if (cfg.backend == CodecBackend::GStreamer) {
#ifdef ENABLE_GSTREAMER
        return std::make_shared<GstDecoder>(cfg);
#else
        return nullptr;  // 请求的 GStreamer 后端未启用：返回空，由上层回退/报错，而非静默给 FFmpeg
#endif
    }
    return std::make_shared<FfmpegDecoder>(cfg);
}

std::shared_ptr<EncoderBackend> create_encoder_backend(const VideoEncoderConfig& cfg) {
    if (cfg.backend == CodecBackend::GStreamer) {
#ifdef ENABLE_GSTREAMER
        return std::make_shared<GstEncoder>(cfg);
#else
        return nullptr;  // 同上：GStreamer 未启用时不回退 FFmpeg
#endif
    }
    return std::make_shared<FfmpegEncoder>(cfg);                 // Task4 FFmpeg 软编
}

} // namespace modeldeploy::video
