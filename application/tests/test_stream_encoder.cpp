#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include "stream_encoder.hpp"
#include "csrc/vision/common/image_data.h"
#ifdef WITH_GPU
#include <cuda_runtime.h>
#include "csrc/vision/processors/cuda/bgr_to_nv12.cuh"
#endif

using namespace modeldeploy::vision;

static EncoderConfig make_cfg() { EncoderConfig c; c.fps = 25; return c; }

TEST_CASE("Encoder construct/destroy", "[encoder]") {
    StreamEncoder enc(make_cfg());
    REQUIRE_FALSE(enc.is_open());
}

TEST_CASE("Encoder open/close cycle", "[encoder]") {
    StreamEncoder enc(make_cfg());
    REQUIRE_NOTHROW(enc.close());
}

TEST_CASE("Encoder double close", "[encoder]") {
    StreamEncoder enc(make_cfg());
    REQUIRE_NOTHROW(enc.close());
    REQUIRE_NOTHROW(enc.close());
}

TEST_CASE("Encoder async without open", "[encoder]") {
    StreamEncoder enc(make_cfg());
    REQUIRE_FALSE(enc.start_async());
    REQUIRE_NOTHROW(enc.stop_async());
}

TEST_CASE("Encoder encode without open", "[encoder]") {
    StreamEncoder enc(make_cfg());
    ImageData img(100, 100, MdImageType::PKG_BGR_U8);
    REQUIRE_FALSE(enc.encode(img));
}

TEST_CASE("Encoder default config", "[encoder]") {
    StreamEncoder enc;
    REQUIRE_FALSE(enc.is_open());
}

#ifdef WITH_GPU
TEST_CASE("bgr_to_nv12_cuda produces valid NV12", "[stream_encoder][gpu]") {
    const int w = 640, h = 360;
    std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3, 128);  // 灰
    uint8_t* d_bgr = nullptr;
    REQUIRE(cudaMalloc(&d_bgr, bgr.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_bgr, bgr.data(), bgr.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    uint8_t* d_nv12 = nullptr;
    REQUIRE(cudaMalloc(&d_nv12, bgr.size() / 2) == cudaSuccess);
    REQUIRE(modeldeploy::vision::bgr_to_nv12_cuda(d_bgr, w, h, d_nv12));
    std::vector<uint8_t> nv12(bgr.size() / 2);
    REQUIRE(cudaMemcpy(nv12.data(), d_nv12, nv12.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    // 灰色 (128,128,128) → Y≈126 (BT.709: (220*128)>>8+16 = 126)，UV 应为中性 128
    REQUIRE(std::abs(static_cast<int>(nv12[0]) - 126) <= 3);
    REQUIRE(std::abs(static_cast<int>(nv12[static_cast<size_t>(w) * h]) - 128) <= 3);
    cudaFree(d_bgr);
    cudaFree(d_nv12);
}
#endif

