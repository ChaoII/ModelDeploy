#include "vision/processors/cuda/cjk_font_util.h"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <vector>
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
