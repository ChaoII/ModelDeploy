#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
#include <algorithm>
#include <fstream>

using namespace modeldeploy::video;

TEST_CASE("GStreamer 软解 h264 → VideoFrame(NV12)", "[video][gst][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) {
        SKIP("no test clip; generate with ffmpeg or place at test_data/video/clip.h264");
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    auto dec = VideoDecoder::create(cfg);
    REQUIRE(dec != nullptr);
    std::string err;
    REQUIRE(dec->open("test_data/video/clip.h264", &err));
    REQUIRE(dec->width() > 0);
    REQUIRE(dec->height() > 0);
    int n = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err) && n < 30) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == dec->width());
        REQUIRE(f.image.height() == dec->height());
        ++n;
    }
    REQUIRE(n > 0);
    dec->close();
}

TEST_CASE("GStreamer qsvh264dec 硬解 → CPU NV12", "[video][gst][hw][gpu][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) SKIP("no test clip; place at test_data/video/clip.h264");
    auto cap = query_video_capabilities();
    bool qsv = std::find(cap.hw_decoders.begin(), cap.hw_decoders.end(), "qsvh264dec") !=
               cap.hw_decoders.end();
    if (!qsv) SKIP("no qsvh264dec decoder in this environment");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Qsv;
    auto dec = VideoDecoder::create(cfg);
    REQUIRE(dec != nullptr);
    std::string err;
    REQUIRE(dec->open("test_data/video/clip.h264", &err));
    REQUIRE(dec->width() > 0);
    REQUIRE(dec->height() > 0);
    int n = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err) && n < 30) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == dec->width());
        REQUIRE(f.image.height() == dec->height());
        ++n;
    }
    REQUIRE(n > 0);
    dec->close();
}
