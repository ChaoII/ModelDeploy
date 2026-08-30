//
// 多路 clone 吞吐基准：单线程 vs N 路 clone 各跑固定帧数，用 TimerArray::sum_ms() 量化。
// 对应 docs/multi_thread.md 的期望：多路 clone 可同时处理多路视频流（每路各自预处+推理）。
// 仅作对照基准（不设置硬性 FPS 断言，避免在无 GPU/弱 GPU 上误报）。
// 运行: ./test_modeldeploy "[benchmark][gpu]clone*"
//
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "csrc/vision.h"
#include "csrc/utils/benchmark.h"

using namespace modeldeploy::vision;

namespace fs = std::filesystem;

namespace {
    fs::path bm_root() {
        const char* d = std::getenv("TEST_DATA_DIR");
        return d ? fs::path(d) / "test_data" : fs::path("test_data");
    }
} // namespace

TEST_CASE("clone multithread throughput", "[benchmark][gpu]") {
    const fs::path root = bm_root();
    const fs::path model = root / "test_models/onnx/yolo26n/yolo26n.onnx";
    if (!fs::exists(model)) { std::cout << "skip: no yolo26n.onnx" << std::endl; return; }

    std::vector<std::string> names = {
        "test_detection0.jpg", "bus.jpg", "test_obb1.jpg", "test_person.jpg",
    };
    std::vector<modeldeploy::vision::ImageData> images;
    for (const auto& n : names) {
        auto im = modeldeploy::vision::ImageData::imread((root / "test_images" / n).string());
        if (!im.empty()) images.push_back(std::move(im));
    }
    if (images.size() < 2) { std::cout << "skip: fewer than 2 images" << std::endl; return; }
    const int nimg = static_cast<int>(images.size());

    modeldeploy::RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_ort_backend();
    modeldeploy::vision::detection::UltralyticsDet det(model.string(), opt);
    if (!det.is_initialized()) { std::cout << "skip: yolo26n.onnx init failed (GPU/ORT)" << std::endl; return; }
    det.get_preprocessor().set_size({640, 640});

    constexpr int kThreads = 4;
    const int total = 200;

    // 预热：加载设备缓冲 + ORT session 首帧开销
    std::vector<modeldeploy::vision::DetectionResult> r;
    for (int i = 0; i < 10; ++i)
        det.predict(images[i % nimg], &r, nullptr);

    // ── 单线程：顺序处理 total 帧 ──
    TimerArray ts;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < total; ++i)
        det.predict(images[i % nimg], &r, &ts);
    auto t1 = std::chrono::steady_clock::now();
    const double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double single_fps = 1000.0 * total / single_ms;
    std::cout << "[bm-clone] single: " << total << " frames in " << single_ms
              << " ms -> " << single_fps << " fps  (TimerArray::sum_ms="
              << ts.sum_ms() << ")" << std::endl;

    // ── 多线程：每路 clone，各自处理 ──
    std::vector<std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet>> clones;
    for (int t = 0; t < kThreads; ++t) clones.push_back(det.clone());
    const int per = total / kThreads;

    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    std::vector<TimerArray> thr_timers(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<modeldeploy::vision::DetectionResult> rr;
            for (int i = 0; i < per; ++i)
                clones[t]->predict(images[(t + i) % nimg], &rr, &thr_timers[t]);
        });
    }
    for (auto& th : threads) th.join();
    auto t3 = std::chrono::steady_clock::now();
    const double mt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double mt_fps = 1000.0 * total / mt_ms;
    TimerArray tmt;
    for (const auto& tt : thr_timers) tmt += tt;
    std::cout << "[bm-clone] multithread(" << kThreads << "): " << total
              << " frames (" << kThreads << "x" << per << ") in " << mt_ms
              << " ms -> " << mt_fps << " fps total, " << (mt_fps / kThreads)
              << " fps/stream  (TimerArray::sum_ms=" << tmt.sum_ms()
              << ", speedup=" << (mt_fps / single_fps) << "x)" << std::endl;

    REQUIRE(true);
}
