#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include "draw_engine.hpp"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/processors/cuda/draw_gpu.cuh"

using namespace modeldeploy::vision;

TEST_CASE("DrawEngine construct", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
}

TEST_CASE("DrawEngine draw on empty results", "[draw]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
    ImageData img(100, 100, MdImageType::PKG_BGR_U8);
    std::vector<InferResult> results;
    REQUIRE_NOTHROW(de.draw(img, results));
}

TEST_CASE("DrawEngine draw with detection results", "[draw]") {
    DrawConfig cfg;
    cfg.show_label = true;
    cfg.show_score = true;
    DrawEngine de(cfg);
    ImageData img(200, 200, MdImageType::PKG_BGR_U8);

    InferResult r;
    r.model_name = "det";
    r.type = "detection";
    DetectionBox box;
    box.x = 10; box.y = 10; box.w = 50; box.h = 50;
    box.score = 0.95f;
    box.label_id = 1;
    r.boxes.push_back(box);

    std::vector<InferResult> results = {r};
    REQUIRE_NOTHROW(de.draw(img, results));

    // 确认像素被修改了
    auto* data = img.data();
    bool has_nonzero = false;
    size_t total = static_cast<size_t>(200) * 200 * 3;
    for (size_t i = 0; i < total; ++i) {
        if (data[i] != 0) { has_nonzero = true; break; }
    }
    REQUIRE(has_nonzero);
}

#ifdef WITH_GPU
TEST_CASE("draw_boxes_gpu modifies interior pixels", "[draw_engine][gpu]") {
    constexpr int W = 640, H = 640;
    std::vector<uint8_t> bgr(W * H * 3, 128);
    uint8_t* d_bgr = nullptr;
    REQUIRE(cudaMalloc(&d_bgr, bgr.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_bgr, bgr.data(), bgr.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    GpuDrawBox box{};
    box.x1 = 100; box.y1 = 100; box.x2 = 200; box.y2 = 200;
    box.score = 0.9f;
    box.label_id = 0;
    box.r = 255; box.g = 0; box.b = 0;   // BGR 红
    std::snprintf(box.label, sizeof(box.label), "person");
    GpuDrawBox* d_box = nullptr;
    REQUIRE(cudaMalloc(&d_box, sizeof(GpuDrawBox)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_box, &box, sizeof(box), cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(modeldeploy::vision::draw_boxes_gpu(d_bgr, W, H, d_box, 1, 0.15f));

    std::vector<uint8_t> out(bgr.size());
    REQUIRE(cudaMemcpy(out.data(), d_bgr, bgr.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    // box 内部像素应被混合（不再是纯 128）
    const int px = (150 * W + 150) * 3;
    REQUIRE_FALSE(out[px] == 128);
    // 边框像素应为纯 box 色
    const int border_px = (100 * W + 100) * 3;
    REQUIRE(out[border_px + 0] == 0);
    REQUIRE(out[border_px + 2] == 255);
    cudaFree(d_bgr);
    cudaFree(d_box);
}

TEST_CASE("DrawEngine::draw_gpu on device BGR", "[draw_engine][gpu]") {
    DrawConfig cfg;
    DrawEngine de(cfg);
    ImageData img(200, 200, MdImageType::PKG_BGR_U8);
    std::memset(img.data(), 0, img.bytes());   // 确定性黑底

    InferResult r;
    r.model_name = "det";
    r.type = "detection";
    DetectionBox box;
    box.x = 10; box.y = 10; box.w = 50; box.h = 50;
    box.score = 0.95f;
    box.label_id = 1;   // 调色板绿色 (BGR 0,255,0)
    box.label_name = "person";
    r.boxes.push_back(box);

    std::vector<InferResult> results = {r};
    REQUIRE(de.draw_gpu(img, results));

    // 边框像素应为纯 box 色
    auto* data = img.data();
    const int bpx = (10 * 200 + 10) * 3;
    REQUIRE(data[bpx + 0] == 0);
    REQUIRE(data[bpx + 1] == 255);
    REQUIRE(data[bpx + 2] == 0);
    // 内部像素应被混合（绿色 α=0.15 叠加黑底 → G≈38）
    const int ipx = (30 * 200 + 30) * 3;
    REQUIRE(data[ipx + 1] >= 30);
}
#endif
