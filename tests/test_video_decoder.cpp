#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <sys/stat.h>
#include "csrc/video/video_decoder.h"

using namespace modeldeploy::video;
using namespace modeldeploy::vision;

static bool file_exists(const char* p) {
    if (!p || !*p) return false;
    struct stat st;
    return ::stat(p, &st) == 0;
}

TEST_CASE("VideoDecoder opens and grabs NV12 frames", "[video]") {
    const char* p = std::getenv("MD_TEST_VIDEO");
    if (!file_exists(p)) {
        SKIP("no test video; set MD_TEST_VIDEO to a local mp4");
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    auto dec = VideoDecoder::create(cfg);
    REQUIRE(dec != nullptr);
    std::string err;
    REQUIRE(dec->open(p, &err));
    REQUIRE(dec->width() > 0);
    REQUIRE(dec->height() > 0);
    REQUIRE(dec->fps() > 0);

    VideoFrame frame;
    int got = 0;
    while (got < 5 && dec->read_one_frame(&frame, &err)) {
        REQUIRE_FALSE(frame.image.empty());
        REQUIRE(frame.image.width() == dec->width());
        REQUIRE(frame.image.height() == dec->height());
        ++got;
    }
    REQUIRE(got > 0);           // 至少抽到一帧
    REQUIRE(frame.image.plane_count() >= 1);
    dec->close();
}
