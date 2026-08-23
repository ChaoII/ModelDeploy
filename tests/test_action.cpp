#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>

#include "csrc/vision/action/tsn.h"
#include "csrc/core/tensor.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::action;

// 无权重路径：构造应 is_initialized()==false，不崩溃。
TEST_CASE("TSN construction without weights", "[action]") {
    TSN m("nonexistent_tsn.onnx");
    REQUIRE_FALSE(m.is_initialized());
}

// preprocess 组装缝：3 帧合成图 → [1, 9, H, W] uni-dim 张量，逐帧 CHW 拼接。
TEST_CASE("TSN assemble_frames builds [1,3T,H,W]", "[action]") {
    const int64_t H = 4, W = 4, T = 3;
    std::vector<ImageData> frames;
    for (int t = 0; t < T; ++t) {
        // RGB 合成图（Packed），恒定像素，便于校验数值
        std::vector<uint8_t> pixels(static_cast<size_t>(H) * W * 3, uint8_t(100));
        frames.emplace_back(ImageData::from_raw(pixels.data(), static_cast<int>(H), static_cast<int>(W),
                                                MdImageType::PKG_RGB_U8, true));
    }
    Tensor out;
    REQUIRE(TSN::assemble_frames(frames, T, H, W, &out));
    REQUIRE(out.get_rank() == 4);
    REQUIRE(out.shape() == std::vector<int64_t>({1, 3 * T, H, W}));
    // 首元素应为 100/255 ≈ 0.392
    const float* p = static_cast<const float*>(out.data());
    REQUIRE(p[0] == Catch::Approx(100.0f / 255.0f).margin(1e-5f));
}

// postprocess：喂合成 [1, C] scores Tenor → argmax label。
TEST_CASE("TSN postprocess sanity", "[action]") {
    TSN m("nonexistent_tsn.onnx");           // 未初始化，仅测纯逻辑后处理
    std::vector<float> vals = {0.1f, 0.7f, 0.2f};
    std::vector<int64_t> shape = {1, 3};
    std::vector<Tensor> outs;
    outs.emplace_back(vals.data(), shape, DataType::FP32, Device::CPU);
    std::vector<float> scores;
    REQUIRE(m.postprocess(outs, &scores));
    REQUIRE(scores.size() == 3);
    // postprocess 仅原样透传 scores（argmax 由 CAPI/上层取）。校验透传。
    REQUIRE(scores[1] == Catch::Approx(0.7f).margin(1e-5f));
}
