//
// 第一批 bug 修复的单元测试
// Bug1: OBB 带 NMS 角度取错索引
// Bug4: OCR un_clip soln 索引
// Bug8: Runtime bind_input/bind_output 共享语义
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <string>
#include <vector>
#include <cstring>
#include "core/tensor.h"
#include "core/enum_variables.h"
#include "vision/obb/postprocessor.h"
#include "vision/common/result.h"
#include "runtime/runtime.h"
#include "runtime/runtime_option.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::detection;

// ============ Bug1: OBB 带 NMS 角度取错索引 ============
// 输出布局 (dim2=7): [xc, yc, w, h, score, label_id, angle(弧度)]
TEST_CASE("OBB with NMS uses correct angle index", "[bugfix]") {
    UltralyticsObbPostprocessor post;
    post.set_conf_threshold(0.25f);
    post.set_nms_threshold(0.7f);

    // 构造 [1, 1, 7] 单框（300 候选太浪费，1 个即可触发 run_with_nms）
    const int dim2 = 7;
    std::vector<float> data(1 * dim2);
    data[0] = 100.0f;  // xc
    data[1] = 100.0f;  // yc
    data[2] = 50.0f;   // w
    data[3] = 30.0f;   // h
    data[4] = 0.9f;    // score（正确索引）
    data[5] = 3.0f;    // label_id
    data[6] = 0.5f;    // angle = 0.5 rad（若被误当 angle，结果会异常）

    Tensor t;
    t.allocate({1, 1, dim2}, DataType::FP32);
    std::memcpy(t.data(), data.data(), data.size() * sizeof(float));

    std::vector<std::vector<ObbResult>> results;
    LetterBoxRecord lb;
    lb.ipt_w = 640; lb.ipt_h = 640; lb.out_w = 640; lb.out_h = 640;
    lb.pad_w = 0; lb.pad_h = 0; lb.scale = 1.0f;

    bool ok = post.run_with_nms({t}, &results, {lb});
    REQUIRE(ok);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].size() == 1);
    const auto& box = results[0][0].rotated_box;
    // 修复前会用 label_id=3 当 angle → 3*180/π ≈ 171.9°
    // 修复后应为 0.5*180/π ≈ 28.6°
    REQUIRE(box.angle == Catch::Approx(0.5f * 180.0f / 3.141592653f));
    REQUIRE(results[0][0].label_id == int32_t{3});
    REQUIRE(results[0][0].score == Catch::Approx(0.9f));
}

// ============ Bug8: Runtime bind_output 共享语义 ============
TEST_CASE("Runtime bind output shares external buffer", "[bugfix]") {
    // bind_output_tensor 只操作内部成员 vector，不依赖后端初始化
    RuntimeOption option;
    Runtime rt;

    std::vector<float> out_buf(16, 1.0f);
    Tensor out_t;
    out_t.from_external_memory(out_buf.data(), {4, 4}, DataType::FP32);
    out_t.set_name("out0");

    rt.bind_output_tensor("out0", out_t);
    auto* bound = rt.get_output_tensor("out0");
    REQUIRE(bound != nullptr);
    // 共享语义：Tensor 指向外部 buffer
    REQUIRE(bound->data() == out_buf.data());
    // 外部修改应反映
    out_buf[0] = 55.0f;
    REQUIRE(static_cast<float*>(bound->data())[0] == 55.0f);
}

