#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>
#include "video_source.hpp"
#include "csrc/video/video_frame.h"

TEST_CASE("VideoSource decodes local file CPU NV12", "[video_source]") {
    const std::string url = "E:/CLionProjects/ModelDeploy/test_data/bench_videos/cam00.mp4";
    if (!std::ifstream(url).good()) { SKIP("test data not present"); }
    REQUIRE(std::ifstream(url).good());

    VideoSource src;
    std::string err;
    REQUIRE(src.open(url, DecoderConfig{}, &err));

    std::atomic<int> frames{0};
    std::atomic<bool> got_nv12{false};
    std::atomic<int> first_w{0}, first_h{0}, first_planes{0};
    src.set_callback([&](modeldeploy::video::VideoFrame&& f) {
        if (f.image.type() == MdImageType::NV12 && !f.image.empty()) {
            if (!got_nv12.exchange(true)) {
                first_w.store((int)f.image.width());
                first_h.store((int)f.image.height());
                first_planes.store((int)f.image.plane_count());
            }
        }
        frames.fetch_add(1);
    });
    REQUIRE(src.start(&err));

    for (int i = 0; i < 200 && frames.load() < 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    src.stop();
    REQUIRE(frames.load() >= 1);
    REQUIRE(got_nv12.load());
    REQUIRE(first_w.load() > 0);
    REQUIRE(first_h.load() > 0);
    REQUIRE(first_planes.load() >= 2);
}
