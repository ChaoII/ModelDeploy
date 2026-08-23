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
    VideoDecoder dec;
    REQUIRE(dec.open(p));
    REQUIRE(dec.width() > 0);
    REQUIRE(dec.height() > 0);
    REQUIRE(dec.fps() > 0.0);

    ImageData frame;
    uint64_t pts = 0;
    int got = 0;
    while (got < 5 && dec.next(&frame, &pts)) {
        REQUIRE_FALSE(frame.empty());
        REQUIRE(frame.width() == dec.width());
        REQUIRE(frame.height() == dec.height());
        ++got;
    }
    REQUIRE(got > 0);           // 至少抽到一帧
    REQUIRE(frame.plane_count() >= 1);
    dec.close();
}
