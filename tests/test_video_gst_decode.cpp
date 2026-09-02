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

TEST_CASE("GStreamer qsvh265dec 硬解(HEVC) → CPU NV12", "[video][gst][hw][gpu][integration]") {
    // GStreamer QSV 解码路径与 H.264(clip.h264) 一致，假定裸素流（无 demux）：
    // filesrc ! h265parse ! qsvh265dec。用 clip.h265（FFmpeg libx265 产出的裸 HEVC 素流）。
    std::ifstream probe("test_data/video/clip.h265");
    if (!probe.good()) SKIP("no HEVC clip; place at test_data/video/clip.h265");
    auto cap = query_video_capabilities();
    bool qsv265dec = std::find(cap.hw_decoders.begin(), cap.hw_decoders.end(), "qsvh265dec") !=
                     cap.hw_decoders.end();
    if (!qsv265dec) SKIP("no qsvh265dec decoder in this environment");
    // 显式 Qsv + codec=hevc_qsv：必须真走 qsvh265dec（HEVC），非 qsvh264dec
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Qsv;
    cfg.codec = "hevc_qsv";
    auto dec = VideoDecoder::create(cfg);
    REQUIRE(dec != nullptr);
    std::string err;
    REQUIRE(dec->open("test_data/video/clip.h265", &err));
    REQUIRE(dec->width() > 0);
    REQUIRE(dec->height() > 0);
    int n = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err) && n < 30) {
        REQUIRE_FALSE(f.image.empty());
        ++n;
    }
    CAPTURE(err);
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
