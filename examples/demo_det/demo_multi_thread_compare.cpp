//
// Created by aichao on 2025/2/24.
// 多线程并发推理对比 demo：
//   方案A（错误）: 同一实例多线程并发 → 触发锁，串行
//   方案B（正确）: clone 后各线程独立实例 → 真正并行
//

#include "csrc/vision.h"
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>

using namespace modeldeploy::vision;

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(10);
    option.use_ort_backend();
    option.use_gpu(0);
    option.enable_fp16 = true;
    option.enable_trt = false;
    option.ort_option.trt_engine_cache_path = "./trt_engine";

    // 加载模型
    modeldeploy::vision::detection::UltralyticsDet yolo11_det(
        "../../test_data/test_models/yolo11n_nms.onnx", option);
    yolo11_det.get_preprocessor().use_cuda_preproc();
    yolo11_det.get_preprocessor().set_size({640, 640});

    // 读取图像
    auto img = modeldeploy::vision::ImageData::imread(
        "../../test_data/test_images/test_detection0.jpg");

    // Warm-up
    std::vector<modeldeploy::vision::DetectionResult> result;
    for (int i = 0; i < 3; ++i) {
        yolo11_det.predict(img, &result);
    }

// 方案A 共享同一实例在多线程下即使有 mutex 保护，
// CUDA kernel 仍可能因 ORT 异步执行而冲突。
// 这里仅用极少推理演示警告日志，不计时。
constexpr int THREAD_COUNT = 4;
constexpr int LOOP_PER_THREAD = 10;       // 方案B迭代数
constexpr int LOOP_A_PER_THREAD = 1;      // 方案A仅做演示，避免CUDA竞争

    // ════════════════════════════════════════════════════════
    // 先跑方案B（正确的 clone 方式），再跑方案A（错误方式）
    // 因为方案A 可能因 CUDA 竞争导致崩溃
    // ════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════
    // 方案B：clone 后各线程独立实例（正确用法）
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 方案B: clone 多线程独立实例（正确用法） =====" << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet>> clones;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        clones.push_back(yolo11_det.clone());
    }

    std::vector<std::thread> threads_b;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads_b.emplace_back([&clones, t, &img, LOOP_PER_THREAD]() {
            std::vector<modeldeploy::vision::DetectionResult> r;
            for (int i = 0; i < LOOP_PER_THREAD; ++i) {
                clones[t]->predict(img, &r);
            }
        });
    }
    for (auto& th : threads_b) th.join();

    auto t1 = std::chrono::steady_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double fps_b = 1000.0 * THREAD_COUNT * LOOP_PER_THREAD / ms_b;
    std::cout << "  耗时: " << ms_b << "ms"
              << "  总推理: " << THREAD_COUNT * LOOP_PER_THREAD << " 帧"
              << "  等效 FPS: " << fps_b << std::endl;

    // 隔离两组测试
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // ════════════════════════════════════════════════════════
    // 方案A：同一实例多线程并发（错误用法）
    // 同一 OrtBackend 实例的 binding_ 被多线程同时改写 = 数据竞争
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 方案A: 同一实例多线程并发（错误用法）=====" << std::endl;
    std::cout << "  直接跑同一实例多线程会因 binding_ 竞争而崩溃。" << std::endl;

    std::vector<std::thread> threads_a;
    for (int t = 0; t < 2; ++t) {
        threads_a.emplace_back([&yolo11_det, &img, LOOP_A_PER_THREAD]() {
            std::vector<modeldeploy::vision::DetectionResult> r;
            for (int i = 0; i < LOOP_A_PER_THREAD; ++i) {
                try {
                    yolo11_det.predict(img, &r);
                } catch (...) {
                    // 忽略 CUDA 错误
                }
            }
        });
    }
    for (auto& th : threads_a) th.join();

    std::cout << "  方案A结束（如果程序到达这里，说明本次运行没有触发崩溃，但这是未定义行为）" << std::endl;

    // ════════════════════════════════════════════════════════
    // 对比
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 对比 =====" << std::endl;
    std::cout << "  方案A（同一实例并发）: 不安全，CUDA Session 无法共享" << std::endl;
    std::cout << "  方案B（clone 后并发）: " << ms_b << "ms  " << fps_b << " FPS" << std::endl;
    return 0;
}
