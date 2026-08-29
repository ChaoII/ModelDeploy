#include <catch2/catch_test_macros.hpp>
#include "video_sink.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/basic_types.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using modeldeploy::vision::ImageData;

static ImageData make_nv12(int w, int h) {
    auto buf = std::make_shared<std::vector<uint8_t>>((size_t)w * h * 3 / 2, 100);
    ImageData::Plane pl[2] = {{buf->data(), w}, {buf->data() + (size_t)w * h, w}};
    return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU, std::move(buf));
}

TEST_CASE("VideoSink writes small flv from CPU NV12", "[video_sink]") {
    const std::string out = std::filesystem::temp_directory_path().string() + "/md_vsink_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".flv";

    EncoderConfig ec;
    ec.codec = "libx264";
    ec.format = "flv";
    VideoSink sink;
    std::string err;
    REQUIRE(sink.open(out, 320, 240, 25, ec, false, &err));

    for (int i = 0; i < 30; ++i) {
        auto img = make_nv12(320, 240);
        REQUIRE(sink.encode(img, &err));
    }
    sink.close();
    REQUIRE(std::filesystem::file_size(out) > 1000);
}
