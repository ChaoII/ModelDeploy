#include "catch2/catch_test_macros.hpp"
#include "csrc/video/factory.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"

using namespace modeldeploy::video;

TEST_CASE("query_video_capabilities 返回可构造的结构", "[video][factory]") {
    auto cap = query_video_capabilities();
    // 任务3/5 起 FFmpeg 与 GStreamer 后端均已启用
    REQUIRE(cap.ffmpeg_available);
    REQUIRE(cap.gstreamer_available);
    REQUIRE(cap.hw_decoders.empty());
    REQUIRE(cap.hw_encoders.empty());
}

TEST_CASE("create_decoder_backend 对 GStreamer 返回后端(已启用)", "[video][factory]") {
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;  // Task5 已接入 GStreamer 软解
    auto b = create_decoder_backend(cfg);
    REQUIRE(b != nullptr);
}

TEST_CASE("create_encoder_backend 对 FFmpeg 返回后端(已填充)", "[video][factory]") {
    VideoEncoderConfig cfg;
    auto b = create_encoder_backend(cfg);
    REQUIRE(b != nullptr);  // Task4 起 FFmpeg 软编后端返回非空
}
