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
    auto* data = img.plane(0).data;
    bool has_nonzero = false;
    size_t total = static_cast<size_t>(200) * 200 * 3;
    for (size_t i = 0; i < total; ++i) {
        if (data[i] != 0) { has_nonzero = true; break; }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("DrawEngine::draw_gpu CPU NV12", "[draw_engine]") {
    const int W = 200, H = 200;
    std::vector<uint8_t> y(W * H, 128), uv(W * H / 2, 128);
    ImageData::Plane pl[2] = {{y.data(), static_cast<size_t>(W)},
                              {uv.data(), static_cast<size_t>(W)}};
    auto img = ImageData::from_planes(pl, 2, MdImageType::NV12, W, H,
                                      modeldeploy::Device::CPU, {});
    REQUIRE(img.plane_count() == 2);

    DrawConfig cfg;
    cfg.show_label = false;
    DrawEngine de(cfg);
    InferResult r;
    r.model_name = "det";
    r.type = "detection";
    DetectionBox b;
    b.x = 10; b.y = 10; b.w = 50; b.h = 50;
    b.score = 0.95f;
    b.label_id = 1;
    b.label_name = "person";
    r.boxes.push_back(b);
    std::vector<InferResult> results = {r};

    REQUIRE(de.draw_gpu(img, results, false, false));
    // 统一 NV12 绘制：Y 平面顶边框被绘制（不再是纯 128）
    const int bpx = 10 * W + 10;
    REQUIRE(y[static_cast<size_t>(bpx)] != 128);
    // packed 帧无 NV12 内核 → draw_gpu 应拒绝并回退（返回 false）
    ImageData pkg(100, 100, MdImageType::PKG_BGR_U8);
    REQUIRE_FALSE(de.draw_gpu(pkg, results, false, false));
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

TEST_CASE("DrawEngine::draw_gpu device NV12 zero-copy", "[draw_engine][gpu]") {
    const int w = 200, h = 200;
    const size_t ybytes = static_cast<size_t>(w) * h;
    const size_t uvbytes = static_cast<size_t>(w) * (h / 2);
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    REQUIRE(cudaMalloc(&y, ybytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&uv, uvbytes) == cudaSuccess);
    REQUIRE(cudaMemset(y, 128, ybytes) == cudaSuccess);
    REQUIRE(cudaMemset(uv, 128, uvbytes) == cudaSuccess);

    // 包装为 Device::GPU NV12 帧：plane 指针即设备内存，零拷贝
    ImageData::Plane pl[2] = {{y, w}, {uv, w}};
    auto img = ImageData::from_planes(pl, 2, MdImageType::NV12, w, h,
                                      modeldeploy::Device::GPU, {});
    REQUIRE(img.plane_count() == 2);
    REQUIRE(img.device() == modeldeploy::Device::GPU);

    DrawConfig cfg;
    cfg.show_label = false;
    DrawEngine de(cfg);
    InferResult r;
    r.model_name = "det";
    r.type = "detection";
    DetectionBox b;
    b.x = 20; b.y = 20; b.w = 60; b.h = 60;
    b.score = 0.95f;
    b.label_id = 1;
    b.label_name = "person";
    r.boxes.push_back(b);
    std::vector<InferResult> results = {r};

    REQUIRE(de.draw_gpu(img, results, false, false));

    // 读回并确认 Y 平面边框像素被绘制（不再是纯 128）
    std::vector<uint8_t> hy(ybytes);
    REQUIRE(cudaMemcpy(hy.data(), y, ybytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    const int bpx = 20 * w + 20;
    REQUIRE(hy[bpx] != 128);
    // 框外仍保持 128：仅在框区域绘制
    REQUIRE(hy[0] == 128);
    // 帧仍是设备 NV12，plane 指针不变：绘制全程留在设备端，无 H2D/D2H
    REQUIRE(img.device() == modeldeploy::Device::GPU);
    REQUIRE(img.plane(0).data == y);
    REQUIRE(img.plane(1).data == uv);
    cudaFree(y);
    cudaFree(uv);
}
#endif
