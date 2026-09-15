#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"

using modeldeploy::vision::DetectionResult;
using modeldeploy::vision::ImageData;

namespace {

// 构造"假引擎"：InferenceEngine 非多态、没有测试注入点，因此用一次注定失败的 load
// 让引擎记录 cfg_（config() 可用）。is_loaded() 为 false，run_models 会走非检测分支且
// infer() 直接返回 false —— 足以覆盖 COW 换指针与并发只读快照路径，且不依赖任何真实模型文件。
InferGroup::ModelFactory fake_factory() {
    return [](const ModelConfig& mc) -> std::unique_ptr<InferenceEngine> {
        auto e = std::make_unique<InferenceEngine>();
        e->load(mc);   // 路径不存在 → 返回 false，但 cfg_ 已记录
        return e;
    };
}

ModelConfig fake_cfg(const std::string& name, float thr = 0.5f) {
    ModelConfig m;
    m.name = name;
    m.type = "detection";
    m.path = "/nonexistent/" + name + ".onnx";
    m.device = "cpu";
    m.input_size = {8, 8};
    m.confidence_threshold = thr;
    m.interval = 1;
    return m;
}

ImageData tiny_image() {
    std::vector<uint8_t> buf(16 * 16 * 3, 0);
    return ImageData::from_raw(buf.data(), 16, 16, MdImageType::PKG_BGR_U8, true);
}

// 通过 update_model 追加两个假模型（add_model 会校验 is_loaded()，假引擎过不了该校验）
void seed(InferGroup& g) {
    auto f = fake_factory();
    REQUIRE(g.update_model(fake_cfg("m1"), f));
    REQUIRE(g.update_model(fake_cfg("m2"), f));
}

}  // namespace

// 并发：一个线程持续 run_models（只读快照），另一线程反复 update_model 换指针。
// 旧实现是 remove+add 直接改 vector，会触发迭代器失效/UB；COW 后必须无崩溃且状态一致。
TEST_CASE("InferGroup 热替换与并发读取不崩溃", "[agent][hotswap]") {
    InferGroup g;
    seed(g);
    REQUIRE_FALSE(g.empty());

    ImageData img = tiny_image();
    std::atomic<bool> stop{false};
    std::atomic<int> run_frames{0};

    std::thread reader([&] {
        std::vector<std::pair<std::string, std::vector<DetectionResult>>> dets;
        std::vector<std::pair<std::string, InferResult>> non_det;
        while (!stop.load()) {
            g.run_models(img, &dets, &non_det);
            ++run_frames;
        }
    });

    auto f = fake_factory();
    for (int i = 0; i < 60; ++i) {
        REQUIRE(g.update_model(fake_cfg("m2", 0.1f + 0.01f * static_cast<float>(i)), f));
    }
    stop = true;
    reader.join();

    REQUIRE(run_frames.load() > 0);
    const ModelConfig* mc = g.config_of("m2");
    REQUIRE(mc != nullptr);
    REQUIRE(mc->confidence_threshold > 0.6f);   // 最终反映最后一次替换（0.1 + 0.01*59 = 0.69）
}

// 写时复制只替换目标条目：未变更模型的引擎实例（地址）必须保持不变。
TEST_CASE("InferGroup 未变更模型不被重建", "[agent][hotswap]") {
    InferGroup g;
    seed(g);

    const ModelConfig* before = g.config_of("m2");
    REQUIRE(before != nullptr);

    REQUIRE(g.update_model(fake_cfg("m1", 0.9f), fake_factory()));

    REQUIRE(g.config_of("m2") == before);            // m2 引擎实例未重建（地址不变）
    REQUIRE(g.config_of("m1") != nullptr);
    REQUIRE(g.config_of("m1")->confidence_threshold == 0.9f);
}

// fail-safe：新引擎构建失败（factory 为空或返回 nullptr）时返回 false，且旧快照原样保留。
TEST_CASE("InferGroup 更新失败保持旧模型（fail-safe）", "[agent][hotswap]") {
    InferGroup g;
    seed(g);

    const ModelConfig* before = g.config_of("m2");
    REQUIRE(before != nullptr);

    // 无 factory
    REQUIRE_FALSE(g.update_model(fake_cfg("m2", 0.99f), nullptr));
    REQUIRE(g.config_of("m2") == before);
    REQUIRE(g.config_of("m2")->confidence_threshold == 0.5f);

    // factory 返回 nullptr
    auto null_factory = [](const ModelConfig&) -> std::unique_ptr<InferenceEngine> {
        return nullptr;
    };
    REQUIRE_FALSE(g.update_model(fake_cfg("m2", 0.99f), null_factory));
    REQUIRE(g.config_of("m2") == before);

    // 旧模型仍在服务：run_models 正常遍历（假引擎无结果 → 返回 false）
    ImageData img = tiny_image();
    std::vector<std::pair<std::string, InferResult>> non_det;
    REQUIRE_FALSE(g.run_models(img, nullptr, &non_det));
    REQUIRE(non_det.size() == 2);
}
