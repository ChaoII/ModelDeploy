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

namespace {

// Auto 候选序：FFmpeg → GStreamer。仅返回"已编译且运行时可用"的后端；两者都不可用返 nullptr。
// 通过临时实例 + runtime_available() 判定，不改动 runtime_available() 本身语义。
std::shared_ptr<DecoderBackend> pick_available_decoder(const VideoDecoderConfig& cfg) {
#ifdef ENABLE_FFMPEG
    {
        auto d = std::make_shared<FfmpegDecoder>(cfg);
        if (d->runtime_available()) return d;
    }
#endif
#ifdef ENABLE_GSTREAMER
    {
        auto d = std::make_shared<GstDecoder>(cfg);
        if (d->runtime_available()) return d;
    }
#endif
    return nullptr;
}

std::shared_ptr<EncoderBackend> pick_available_encoder(const VideoEncoderConfig& cfg) {
#ifdef ENABLE_FFMPEG
    {
        auto e = std::make_shared<FfmpegEncoder>(cfg);
        if (e->runtime_available()) return e;
    }
#endif
#ifdef ENABLE_GSTREAMER
    {
        auto e = std::make_shared<GstEncoder>(cfg);
        if (e->runtime_available()) return e;
    }
#endif
    return nullptr;
}

}  // namespace

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
    // Auto：候选序 FFmpeg→GStreamer，只选"已编译且运行时可用"者；均不可用返回 nullptr（由门面报后端不可用）
    if (cfg.backend == CodecBackend::Auto)
        return pick_available_decoder(cfg);

    // 显式 GStreamer：已编译且运行时可用→返回；已编译但不可用→返回空（不静默换 FFmpeg）；
    // 未编译（ENABLE_GSTREAMER=OFF）→返回空，保持 Phase1 收尾语义。
    if (cfg.backend == CodecBackend::GStreamer) {
#ifdef ENABLE_GSTREAMER
        auto d = std::make_shared<GstDecoder>(cfg);
        if (d->runtime_available()) return d;
        return nullptr;  // 已编译但运行时不可用：不静默换后端
#else
        return nullptr;  // 请求的 GStreamer 后端未启用：返回空，由上层回退/报错，而非静默给 FFmpeg
#endif
    }
    // 显式 FFmpeg（默认）：已编译且运行时可用→返回；不可用→返回空（不静默换）
#ifdef ENABLE_FFMPEG
    auto d = std::make_shared<FfmpegDecoder>(cfg);
    if (d->runtime_available()) return d;
    return nullptr;
#else
    return nullptr;
#endif
}

std::shared_ptr<EncoderBackend> create_encoder_backend(const VideoEncoderConfig& cfg) {
    if (cfg.backend == CodecBackend::Auto)
        return pick_available_encoder(cfg);

    if (cfg.backend == CodecBackend::GStreamer) {
#ifdef ENABLE_GSTREAMER
        auto e = std::make_shared<GstEncoder>(cfg);
        if (e->runtime_available()) return e;
        return nullptr;
#else
        return nullptr;  // 同上：GStreamer 未启用时不回退 FFmpeg
#endif
    }
#ifdef ENABLE_FFMPEG
    auto e = std::make_shared<FfmpegEncoder>(cfg);
    if (e->runtime_available()) return e;
    return nullptr;
#else
    return nullptr;
#endif
}

} // namespace modeldeploy::video
