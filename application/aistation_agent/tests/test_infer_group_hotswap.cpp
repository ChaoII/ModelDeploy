#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"

using modeldeploy::vision::DetectionResult;
using modeldeploy::vision::ImageData;

namespace {

// 生产契约：InferGroup::update_model 换入前强制 is_loaded() 就绪校验（本任务把校验从控制面
// 入口下沉到此层，覆盖所有调用方）。InferenceEngine 非多态、无测试注入点，且不再接受
// "非空但未就绪"的引擎，因此无法再用注定失败的 load 造假引擎。
// 改用真实 det 模型建原型，factory 通过 clone_detection_from 复用 ORT Session（不重复加载，
// 60 次热更新依然很快）；测试数据缺失时 SKIP（与 test_inference_* 的既有约定一致）。
constexpr const char* kDetPath = "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx";

bool det_data_available() { return std::ifstream(kDetPath).good(); }

ModelConfig real_cfg(const std::string& name, float thr = 0.5f) {
    ModelConfig m;
    m.name = name;
    m.type = "detection";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kDetPath;
    m.labels = {"person"};
    m.input_size = {640, 640};
    m.confidence_threshold = thr;
    m.interval = 1;
    return m;
}

/// 真实 det 原型：只加载一次，作为 clone 源（共享 ORT Session）。
struct DetPrototype {
    InferenceEngine engine;
    bool ok = false;
    DetPrototype() { ok = engine.load(real_cfg("__proto__")); }
};

/// factory：从原型 clone（is_loaded()==true，满足新的就绪契约）
InferGroup::ModelFactory clone_factory(InferenceEngine* proto) {
    return [proto](const ModelConfig& mc) -> std::unique_ptr<InferenceEngine> {
        auto e = std::make_unique<InferenceEngine>();
        e->clone_detection_from(*proto->det_model(), mc);
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

/// 通过 update_model 追加两个真实模型
void seed(InferGroup& g, InferenceEngine* proto) {
    auto f = clone_factory(proto);
    REQUIRE(g.update_model(real_cfg("m1"), f));
    REQUIRE(g.update_model(real_cfg("m2"), f));
}

}  // namespace

// 并发：一个线程持续 run_models（只读快照），另一线程反复 update_model 换指针。
// 旧实现是 remove+add 直接改 vector，会触发迭代器失效/UB；COW 后必须无崩溃且状态一致。
TEST_CASE("InferGroup 热替换与并发读取不崩溃", "[agent][hotswap]") {
    if (!det_data_available()) SKIP("test data absent");
    DetPrototype proto;
    REQUIRE(proto.ok);
    InferGroup g;
    seed(g, &proto.engine);
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

    auto f = clone_factory(&proto.engine);
    for (int i = 0; i < 60; ++i) {
        REQUIRE(g.update_model(real_cfg("m2", 0.1f + 0.01f * static_cast<float>(i)), f));
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
    if (!det_data_available()) SKIP("test data absent");
    DetPrototype proto;
    REQUIRE(proto.ok);
    InferGroup g;
    seed(g, &proto.engine);

    // 生命周期约束：config_of() 返回快照内裸指针，仅在"该模型未被替换"期间有效。
    // 本用例后续只替换 m1，故 before(m2) 在替换后仍指向原引擎（写时复制保留 m2 条目）。
    const ModelConfig* before = g.config_of("m2");
    REQUIRE(before != nullptr);

    REQUIRE(g.update_model(real_cfg("m1", 0.9f), clone_factory(&proto.engine)));

    REQUIRE(g.config_of("m2") == before);            // m2 引擎实例未重建（地址不变）
    REQUIRE(g.config_of("m1") != nullptr);
    REQUIRE(g.config_of("m1")->confidence_threshold == 0.9f);
}

// fail-safe：
//  a) 无 factory / factory 返回 nullptr → false，旧快照原样保留；
//  b) factory 返回"非空但未就绪"引擎 → 被就绪校验拒绝（本任务下沉点），旧快照保留。
TEST_CASE("InferGroup 更新失败保持旧模型（fail-safe）", "[agent][hotswap]") {
    if (!det_data_available()) SKIP("test data absent");
    DetPrototype proto;
    REQUIRE(proto.ok);
    InferGroup g;
    seed(g, &proto.engine);

    const ModelConfig* before = g.config_of("m2");
    REQUIRE(before != nullptr);

    // 无 factory
    REQUIRE_FALSE(g.update_model(real_cfg("m2", 0.99f), nullptr));
    REQUIRE(g.config_of("m2") == before);
    REQUIRE(g.config_of("m2")->confidence_threshold == 0.5f);

    // factory 返回 nullptr
    auto null_factory = [](const ModelConfig&) -> std::unique_ptr<InferenceEngine> {
        return nullptr;
    };
    REQUIRE_FALSE(g.update_model(real_cfg("m2", 0.99f), null_factory));
    REQUIRE(g.config_of("m2") == before);

    // factory 返回"非空但未就绪"引擎（is_loaded()==false，如 create_engine 对 cls/ocr/lpr
    // 加载失败仍返回非空）→ 必须被就绪校验拒绝，绝不换入（fail-safe）
    auto not_ready = [](const ModelConfig& mc) -> std::unique_ptr<InferenceEngine> {
        auto e = std::make_unique<InferenceEngine>();
        e->load(mc);   // 路径不存在 → is_loaded()==false，但返回的引擎非空
        return e;
    };
    REQUIRE_FALSE(g.update_model(fake_cfg("m2", 0.99f), not_ready));
    REQUIRE(g.config_of("m2") == before);
    REQUIRE(g.config_of("m2")->confidence_threshold == 0.5f);

    // 旧模型仍在服务：run_models 正常遍历到 2 个 detection 模型（结果走 sdk_dets 通道）
    ImageData img = tiny_image();
    std::vector<std::pair<std::string, std::vector<DetectionResult>>> dets;
    std::vector<std::pair<std::string, InferResult>> non_det;
    g.run_models(img, &dets, &non_det);
    REQUIRE(dets.size() == 2);
    REQUIRE(non_det.empty());
}

// 并发审计 HIGH：旧 API config_of()/det_model() 返回快照内裸指针，函数返回即可能悬垂
// （并发热更新/移除换出新快照后旧引擎被释放 → UAF）。find_model() 返回的 ModelRef 以
// 强引用保活引擎：即使当前快照已换出该条目，旧句柄的解引用依然有效（值不变）。
TEST_CASE("InferGroup ModelRef 跨热更新保活（防悬垂 UAF）", "[agent][hotswap]") {
    if (!det_data_available()) SKIP("test data absent");
    DetPrototype proto;
    REQUIRE(proto.ok);
    InferGroup g;
    seed(g, &proto.engine);

    // 取旧引擎句柄，随后热更新同一条目换入新引擎
    InferGroup::ModelRef old_ref = g.find_model("m2");
    REQUIRE(static_cast<bool>(old_ref));
    REQUIRE(old_ref.config() != nullptr);
    REQUIRE(old_ref.config()->confidence_threshold == 0.5f);
    REQUIRE(old_ref.det_model() != nullptr);

    REQUIRE(g.update_model(real_cfg("m2", 0.9f), clone_factory(&proto.engine)));

    // 当前快照已换新引擎
    InferGroup::ModelRef new_ref = g.find_model("m2");
    REQUIRE(static_cast<bool>(new_ref));
    REQUIRE(new_ref.config()->confidence_threshold == 0.9f);
    REQUIRE(new_ref.engine().get() != old_ref.engine().get());   // 确实换出

    // 旧句柄仍保活旧引擎：解引用安全且读到旧值（旧实现此处读已释放内存 = UB）
    REQUIRE(old_ref.config() != nullptr);
    REQUIRE(old_ref.config()->confidence_threshold == 0.5f);
    REQUIRE(old_ref.det_model() != nullptr);

    // 显式移除后旧快照失去最后一个快照持有者，但句柄仍保活
    REQUIRE(g.remove_model("m2"));
    REQUIRE_FALSE(static_cast<bool>(g.find_model("m2")));
    REQUIRE(old_ref.config() != nullptr);
    REQUIRE(old_ref.config()->confidence_threshold == 0.5f);
    REQUIRE(old_ref.det_model() != nullptr);
}
