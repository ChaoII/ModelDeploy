#include "catch2/catch_test_macros.hpp"
#include "csrc/video/factory.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"

using namespace modeldeploy::video;

TEST_CASE("query_video_capabilities 返回可构造的结构", "[video][factory]") {
    auto cap = query_video_capabilities();
    // 后端可用性由 cfg 之外的全局开关推断；此处只断言结构可构造、字段可读（骨架默认均为 false/空）
    REQUIRE_FALSE(cap.ffmpeg_available);
    REQUIRE_FALSE(cap.gstreamer_available);
    REQUIRE(cap.hw_decoders.empty());
    REQUIRE(cap.hw_encoders.empty());
}

TEST_CASE("create_decoder_backend 对 GStreamer 返回空(未启用)", "[video][factory]") {
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;  // Phase1 基线 ENABLE_GSTREAMER=OFF
    auto b = create_decoder_backend(cfg);
    // 当 GSTREAMER 后端未编译时该路径返回 nullptr（工厂探测逻辑）
    if (!query_video_capabilities().gstreamer_available) {
        REQUIRE(b == nullptr);
    }
}

TEST_CASE("create_encoder_backend 对 FFmpeg 返回空(骨架未填充)", "[video][factory]") {
    VideoEncoderConfig cfg;
    auto b = create_encoder_backend(cfg);
    REQUIRE(b == nullptr);  // Task3-6 填充真实后端前，骨架一律返回空
}
