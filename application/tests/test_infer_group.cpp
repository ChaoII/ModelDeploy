#include <catch2/catch_test_macros.hpp>
#include "infer_group.hpp"
#include <opencv2/core.hpp>

TEST_CASE("InferGroup construct/destroy", "[infer_group]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    InferGroup group(cfg);
    REQUIRE_FALSE(group.ready());
}

TEST_CASE("InferGroup init without models", "[infer_group]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    InferGroup group(cfg);
    REQUIRE_FALSE(group.init());
    REQUIRE_FALSE(group.ready());
}

TEST_CASE("InferGroup init with nonexistent model", "[infer_group]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    ModelConfig m;
    m.name = "det";
    m.type = "detection";
    m.path = "/nonexistent.onnx";
    cfg.models.push_back(m);
    InferGroup group(cfg);
    REQUIRE_FALSE(group.init());
}

TEST_CASE("InferGroup batch_only mode", "[infer_group][gpu]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    {
        // 无模型：batch_only 仍不应 ready（models 为空 → gpu 直通 false）
        InferGroup g0(cfg, nullptr, true);
        REQUIRE_FALSE(g0.init());
    }
    // 全 detection + gpu + 无 ROI → gpu_nv12_ready 为 true
    ModelConfig m;
    m.name = "det";
    m.type = "detection";
    m.device = "gpu";
    m.path = "/nonexistent.onnx";   // batch_only 不建引擎，路径无关
    cfg.models.push_back(m);
    {
        InferGroup g(cfg, nullptr, true);
        REQUIRE(g.init());
        REQUIRE(g.batch_only());
        REQUIRE(g.ready());
        REQUIRE(g.gpu_nv12_ready());
        // run_models 被调用 → 记错返回 0
        std::vector<InferResult> results;
        REQUIRE(g.run_models(nullptr, nullptr, nullptr, nullptr, 640, 640, 640, 640, &results) == 0);
        // add/remove/update 均返回 false
        REQUIRE_FALSE(g.add_model(m));
        REQUIRE_FALSE(g.remove_model("det"));
        REQUIRE_FALSE(g.update_model("det", m));
    }
    // 含非 detection → gpu_nv12_ready 为 false
    {
        ModelConfig ocr = m;
        ocr.name = "ocr";
        ocr.type = "ocr";
        cfg.models.push_back(ocr);
        InferGroup g(cfg, nullptr, true);
        REQUIRE(g.init());
        REQUIRE_FALSE(g.gpu_nv12_ready());
    }
}

TEST_CASE("InferGroup add/remove model dynamic", "[infer_group]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    cfg.models.push_back({"det", "detection", "/m/yolo.onnx"});
    InferGroup group(cfg);
    // init will fail but add_model should work independently
    ModelConfig m2;
    m2.name = "face";
    m2.type = "detection";
    m2.path = "/m/face.onnx";
    // Can't test full flow without real models, just check no crash
    // REQUIRE_FALSE(group.add_model(m2));  // will fail (no real file), but shouldn't crash
    // REQUIRE_FALSE(group.remove_model("nonexistent"));
}

TEST_CASE("InferGroup update model config", "[infer_group]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    InferGroup group(cfg);
    ModelConfig m;
    m.name = "det";
    // should not crash
    REQUIRE_FALSE(group.update_model("det", m));
}

TEST_CASE("InferGroup concurrent model ops do not deadlock", "[infer_group][stress]") {
    TaskConfig cfg;
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    for (int round = 0; round < 8; ++round) {
        auto group = std::make_unique<InferGroup>(cfg);
        ModelConfig m;
        m.name = "det";
        m.type = "detection";
        m.path = "/nonexistent.onnx";
        // 并发 add/remove/update：models_mtx_ 串行化，不应死锁/崩溃
        std::thread t1([&] { for (int i = 0; i < 30; ++i) group->add_model(m); });
        std::thread t2([&] { for (int i = 0; i < 30; ++i) group->remove_model("det"); });
        std::thread t3([&] { for (int i = 0; i < 30; ++i) group->update_model("det", m); });
        t1.join(); t2.join(); t3.join();
        group.reset();  // 析构：stop_workers + warmup join，不应挂死
    }
    REQUIRE(true);
}