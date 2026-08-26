#include "vision/processors/cuda/cjk_font_util.h"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#include "vision/processors/cuda/draw_gpu.cuh"
namespace mv = modeldeploy::vision;
TEST_CASE("cjk_font: lookup and utf8", "[cjkfont][core]") {
    uint32_t cp = 0;
    REQUIRE(mv::utf8_to_cp("A", &cp) == 1);
    REQUIRE(cp == 0x41);
    const char* zi = "\xE4\xB8\xAD";  // "中"
    REQUIRE(mv::utf8_to_cp(zi, &cp) == 3);
    REQUIRE(cp == 0x4E2D);
    REQUIRE(mv::cjk_lookup(0x41) >= 0);
    REQUIRE(mv::cjk_lookup(0x4E2D) >= 0);
}

TEST_CASE("cuda draw fill_rect nv12 (semantic)", "[gpu]") {
    constexpr int w = 64, h = 48;
    std::vector<uint8_t> y_host(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> uv_host(static_cast<size_t>(w) * (h / 2), 128);
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, y_host.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, uv_host.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, y_host.data(), y_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, uv_host.data(), uv_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    // 纯红(255,0,0) BT.601 全幅 → Y = ((66*255+128)>>8)+16 = 82; alpha=1 直写
    REQUIRE(modeldeploy::vision::fill_rect_nv12_gpu(d_y, d_uv, w, h, w, w,
                                                    4, 4, 32, 32, 255, 0, 0, 1.0f, nullptr));
    std::vector<uint8_t> back(y_host.size());
    REQUIRE(cudaMemcpy(back.data(), d_y, y_host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(back[4 * w + 8] == 82);      // 矩形内(8,4) → 82
    REQUIRE(back[0] == 128);             // 矩形外(0,0)不变
    cudaFree(d_y); cudaFree(d_uv);
}

TEST_CASE("cuda draw_line nv12 (semantic)", "[gpu]") {
    constexpr int w = 64, h = 48;
    std::vector<uint8_t> y_host(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> uv_host(static_cast<size_t>(w) * (h / 2), 128);
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, y_host.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, uv_host.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, y_host.data(), y_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, uv_host.data(), uv_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    // 水平线 y=24,纯白(255,255,255) → Y=((66+129+25)*255+128)>>8)+16 = (255+128)>>8=1 ->+16=... 见下
    REQUIRE(modeldeploy::vision::draw_line_nv12_gpu(d_y, d_uv, w, h, w, w,
                                                    0.0f, 24.0f, static_cast<float>(w), 24.0f,
                                                    255, 255, 255, 3, nullptr));
    std::vector<uint8_t> back(y_host.size());
    REQUIRE(cudaMemcpy(back.data(), d_y, y_host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    // 线上点(32,24)应被写(纯白 Y=255+16? 见下公式) → 断言 back[24*w+32] != 128 即为被写
    REQUIRE(back[24 * w + 32] != 128);
    REQUIRE(back[0] == 128);   // 线外不变
    cudaFree(d_y); cudaFree(d_uv);
}

TEST_CASE("cuda vis_pose_nv12 (semantic)", "[gpu]") {
    constexpr int w = 64, h = 48;
    std::vector<uint8_t> y_host(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> uv_host(static_cast<size_t>(w) * (h / 2), 128);
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, y_host.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, uv_host.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, y_host.data(), y_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, uv_host.data(), uv_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    const mv::ImageData::Plane planes[2] = {{d_y, w}, {d_uv, w}};
    mv::ImageData frame = mv::ImageData::from_planes(planes, 2, MdImageType::NV12, w, h, modeldeploy::Device::GPU);

    // 单 person:两可见关键点(左/右肩,skel {6,7})连成一条骨架线段;其余低置信不画
    mv::KeyPointsResult r;
    r.label_id = 0;
    r.score = 0.9f;
    r.box = {8.0f, 8.0f, 16.0f, 16.0f};
    r.keypoints.assign(17, mv::Point3f(0, 0, 0.0f));
    r.keypoints[5] = mv::Point3f(20.0f, 20.0f, 1.0f);   // 左肩(COCO 1-indexed=6)
    r.keypoints[6] = mv::Point3f(30.0f, 20.0f, 1.0f);   // 右肩(COCO 1-indexed=7)
    r.keypoints[10] = mv::Point3f(50.0f, 40.0f, 0.0f);  // 低置信点(框外)
    std::vector<mv::KeyPointsResult> results{r};

    mv::VisionProcessorBackend::VisOptions opt;
    opt.alpha = 1.0;
    opt.threshold = 0.5;

    mv::CudaProcessorBackend backend;
    REQUIRE(backend.vis_pose_nv12(frame, results, opt));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint8_t> back(y_host.size());
    REQUIRE(cudaMemcpy(back.data(), d_y, y_host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(back[12 * w + 12] != 128);      // 框内被填充
    REQUIRE(back[20 * w + 25] != 128);      // 骨架线段中点(25,20)被画
    REQUIRE(back[40 * w + 50] == 128);      // 低置信点不画
    cudaFree(d_y); cudaFree(d_uv);
}

TEST_CASE("cuda draw_text_cjk nv12 (semantic)", "[gpu]") {
    constexpr int w = 64, h = 48;
    std::vector<uint8_t> y_host(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> uv_host(static_cast<size_t>(w) * (h / 2), 128);
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, y_host.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, uv_host.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, y_host.data(), y_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, uv_host.data(), uv_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    const char* txt = "\xE4\xB8\xAD" "A\xE4\xB8\xAD";  // "中A中"
    REQUIRE(modeldeploy::vision::draw_text_cjk_nv12_gpu(d_y, d_uv, w, h, w, w,
                                                        0.0f, 0.0f, txt, 255, 255, 255, 1, 16, nullptr));
    std::vector<uint8_t> back(y_host.size());
    REQUIRE(cudaMemcpy(back.data(), d_y, y_host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    int written = 0;
    for (auto v : back) if (v != 128) ++written;
    REQUIRE(written > 100);   // 至少画出若干字形像素(非空文本)
    // 首字符"中"位于 0..16;msyh 垂直居中致顶部若干行空白,故检查第一个 16x16 单元内是否有被写像素
    bool first_char_written = false;
    for (int y = 0; y < 16 && y < h; ++y)
        for (int x = 0; x < 16 && x < w; ++x)
            if (back[y * w + x] != 128) first_char_written = true;
    REQUIRE(first_char_written);
    cudaFree(d_y); cudaFree(d_uv);
}

TEST_CASE("cuda vis_det_nv12 (semantic)", "[gpu]") {
    constexpr int w = 64, h = 48;
    std::vector<uint8_t> y_host(static_cast<size_t>(w) * h, 128);
    std::vector<uint8_t> uv_host(static_cast<size_t>(w) * (h / 2), 128);
    uint8_t* d_y = nullptr; uint8_t* d_uv = nullptr;
    REQUIRE(cudaMalloc(&d_y, y_host.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_uv, uv_host.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_y, y_host.data(), y_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_uv, uv_host.data(), uv_host.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    // 构造一个 NV12 设备 ImageData(借用 d_y/d_uv,Device::GPU)
    const mv::ImageData::Plane planes[2] = {{d_y, w}, {d_uv, w}};
    mv::ImageData frame = mv::ImageData::from_planes(planes, 2, MdImageType::NV12, w, h, modeldeploy::Device::GPU);

    mv::DetectionResult det;
    det.label_id = 0;
    det.score = 0.9f;
    det.box = {8.0f, 8.0f, 16.0f, 16.0f};
    std::vector<mv::DetectionResult> results{det};

    mv::VisionProcessorBackend::VisOptions opt;
    opt.alpha = 1.0;            // 全幅直写,回读断言确定
    opt.threshold = 0.5;

    mv::CudaProcessorBackend backend;
    REQUIRE(backend.vis_det_nv12(frame, results, opt));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint8_t> back(y_host.size());
    REQUIRE(cudaMemcpy(back.data(), d_y, y_host.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    // 框内像素被填充色改写(label 0 调色板 {230,159,0} → Y≠128)
    REQUIRE(back[12 * w + 16] != 128);
    // 框外保留原值
    REQUIRE(back[0] == 128);
    REQUIRE(back[47 * w + 40] == 128);
    cudaFree(d_y); cudaFree(d_uv);
}
