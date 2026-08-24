#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_frame.h"
#include "csrc/video/video_common.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/adapter.h"

using namespace modeldeploy::video;

TEST_CASE("VideoFrame 内嵌 ImageData + pts", "[video][core]") {
    VideoFrame f;
    REQUIRE(f.image.empty());
    REQUIRE(f.pts_ms == 0);
    f.pts_ms = 1234;
    REQUIRE(f.pts_ms == 1234);
}

TEST_CASE("CodecBackend / HwAccel 字符串往返", "[video][core]") {
    REQUIRE(backend_to_string(CodecBackend::FFmpeg) == "ffmpeg");
    REQUIRE(backend_to_string(CodecBackend::GStreamer) == "gstreamer");
    REQUIRE(hwaccel_to_string(HwAccel::Auto) == "auto");
}

TEST_CASE("VideoEncoderConfig 链式 setter 与 validate", "[video][core]") {
    VideoEncoderConfig cfg;
    cfg.set_fps(30).set_bitrate_kbps(2500).set_codec("libx264").set_format("mp4");
    std::string err;
    REQUIRE(cfg.validate(&err));
    VideoEncoderConfig bad_fps;
    bad_fps.set_fps(0).set_format("mp4");
    REQUIRE_FALSE(bad_fps.validate(&err));

    VideoEncoderConfig bad_format;
    bad_format.set_fps(30).set_format("unknown_fmt");
    REQUIRE_FALSE(bad_format.validate(&err));
}

TEST_CASE("适配器由平面视图构造 NV12 ImageData", "[video][core]") {
    uint8_t y[4]{1,2,3,4};
    uint8_t uv[2]{5,6};
    IPlaneView v{y, 2, uv, 2, 2, 2, modeldeploy::Device::CPU, {}};
    auto img = make_image_from_planes_view(v);
    REQUIRE_FALSE(img.empty());
    REQUIRE(img.width() == 2);
    REQUIRE(img.height() == 2);
    REQUIRE(img.plane_count() == 2);
    REQUIRE(img.plane(0).data == y);
    REQUIRE(img.plane(1).data == uv);
}
