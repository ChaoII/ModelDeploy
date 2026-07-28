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

    constexpr int THREAD_COUNT = 4;
    constexpr int LOOP_PER_THREAD = 10;

    // ════════════════════════════════════════════════════════
    // 方案A：同一实例多线程并发（错误用法）
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 方案A: 同一实例多线程并发（错误用法） =====" << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads_a;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads_a.emplace_back([&yolo11_det, &img, LOOP_PER_THREAD]() {
            std::vector<modeldeploy::vision::DetectionResult> r;
            for (int i = 0; i < LOOP_PER_THREAD; ++i) {
                yolo11_det.predict(img, &r);
            }
        });
    }
    for (auto& th : threads_a) th.join();

    auto t1 = std::chrono::steady_clock::now();
    auto ms_a = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double fps_a = 1000.0 * THREAD_COUNT * LOOP_PER_THREAD / ms_a;
    std::cout << "  耗时: " << ms_a << "ms"
              << "  总推理: " << THREAD_COUNT * LOOP_PER_THREAD << " 帧"
              << "  等效 FPS: " << fps_a << std::endl;

    // ════════════════════════════════════════════════════════
    // 方案B：clone 后各线程独立实例（正确用法）
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 方案B: clone 多线程独立实例（正确用法） =====" << std::endl;
    auto t2 = std::chrono::steady_clock::now();

    // 每个线程 clone 一份
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

    auto t3 = std::chrono::steady_clock::now();
    auto ms_b = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    double fps_b = 1000.0 * THREAD_COUNT * LOOP_PER_THREAD / ms_b;
    std::cout << "  耗时: " << ms_b << "ms"
              << "  总推理: " << THREAD_COUNT * LOOP_PER_THREAD << " 帧"
              << "  等效 FPS: " << fps_b << std::endl;

    // ════════════════════════════════════════════════════════
    // 对比
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 对比 =====" << std::endl;
    std::cout << "  方案A（同一实例并发）: " << ms_a << "ms  " << fps_a << " FPS" << std::endl;
    std::cout << "  方案B（clone 后并发）: " << ms_b << "ms  " << fps_b << " FPS" << std::endl;
    std::cout << "  加速比: " << (double)ms_a / ms_b << "x" << std::endl;
    return 0;
}
