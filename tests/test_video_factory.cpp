#include "catch2/catch_test_macros.hpp"
#include "csrc/video/factory.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"

using namespace modeldeploy::video;

namespace {
// 能力与工厂一致性校验：能力为 true 则对应后端非空，为 false 则必须返回空（不回退/不凭空造出后端）
void check_backend_consistency(const VideoCodecCapabilities& cap,
                               CodecBackend decoder_backend_to_probe,
                               CodecBackend encoder_backend_to_probe,
                               bool expect) {
    VideoDecoderConfig dcfg;
    dcfg.backend = decoder_backend_to_probe;
    auto db = create_decoder_backend(dcfg);
    REQUIRE((db != nullptr) == expect);

    VideoEncoderConfig ecfg;
    ecfg.backend = encoder_backend_to_probe;
    auto eb = create_encoder_backend(ecfg);
    REQUIRE((eb != nullptr) == expect);
}
}  // namespace

TEST_CASE("query_video_capabilities 返回可解析的结构", "[video][factory]") {
    auto cap = query_video_capabilities();
    // 能力位要么是布尔要么已初始化；不硬编码任一后端必须可用，随构建配置变化
    REQUIRE_NOTHROW(static_cast<void>(cap.ffmpeg_available));
    REQUIRE_NOTHROW(static_cast<void>(cap.gstreamer_available));
    REQUIRE(cap.hw_decoders.empty());
    REQUIRE(cap.hw_encoders.empty());
}

TEST_CASE("FFmpeg 后端能力与工厂一致", "[video][factory]") {
    auto cap = query_video_capabilities();
    check_backend_consistency(cap, CodecBackend::FFmpeg, CodecBackend::FFmpeg, cap.ffmpeg_available);
}

TEST_CASE("GStreamer 后端能力与工厂一致", "[video][factory]") {
    auto cap = query_video_capabilities();
    check_backend_consistency(cap, CodecBackend::GStreamer, CodecBackend::GStreamer,
                              cap.gstreamer_available);
}

