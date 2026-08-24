#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_decoder.h"
#include <fstream>

using namespace modeldeploy::video;

TEST_CASE("FFmpeg 软解 h264 → VideoFrame(NV12)", "[video][ffmpeg][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) {
        SKIP("no test clip; generate with ffmpeg or place at test_data/video/clip.h264");
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
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
