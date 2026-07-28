//
// Created by aichao on 2025/2/24.
// 多线程/单线程推理性能对比 demo
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

    // 读取图像（只读一次，所有线程共享图像数据）
    auto img = modeldeploy::vision::ImageData::imread(
        "../../test_data/test_images/test_detection0.jpg");

    // ── 预热 ──
    std::vector<modeldeploy::vision::DetectionResult> warmup_result;
    std::cout << "Warming up ..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        yolo11_det.predict(img, &warmup_result);
    }
    std::cout << "Warm-up done." << std::endl;

    constexpr int LOOP_COUNT = 100;
    constexpr int THREAD_COUNT = 4;

    // ════════════════════════════════════════════════════════
    // 单线程推理
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 单线程推理 =====" << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < LOOP_COUNT; ++i) {
        yolo11_det.predict(img, &warmup_result);
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms_single = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double fps_single = 1000.0 * LOOP_COUNT / ms_single;
    std::cout << "  推理 " << LOOP_COUNT << " 帧"
              << "  耗时 " << ms_single << "ms"
              << "  平均 " << (double)ms_single / LOOP_COUNT << "ms/帧"
              << "  " << fps_single << " FPS" << std::endl;

    // ════════════════════════════════════════════════════════
    // 多线程推理（clone 后各线程独立实例）
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 多线程推理 (" << THREAD_COUNT << " 线程, clone) =====" << std::endl;

    // 提前 clone
    std::vector<std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet>> clones;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        clones.push_back(yolo11_det.clone());
    }

    auto t2 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&clones, t, &img, LOOP_COUNT]() {
            std::vector<modeldeploy::vision::DetectionResult> r;
            for (int i = 0; i < LOOP_COUNT; ++i) {
                clones[t]->predict(img, &r);
            }
        });
    }
    for (auto& th : threads) th.join();

    auto t3 = std::chrono::steady_clock::now();
    auto ms_mt = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    int total_frames_mt = THREAD_COUNT * LOOP_COUNT;
    double fps_mt = 1000.0 * total_frames_mt / ms_mt;
    std::cout << "  推理 " << total_frames_mt << " 帧 (" << THREAD_COUNT
              << " 线程 × " << LOOP_COUNT << " 帧/线程)"
              << "  耗时 " << ms_mt << "ms"
              << "  等效 " << fps_mt << " FPS" << std::endl;

    // ════════════════════════════════════════════════════════
    // 对比
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 对比 =====" << std::endl;
    std::cout << "  单线程:            " << fps_single << " FPS" << std::endl;
    std::cout << "  多线程 (" << THREAD_COUNT << " clone): " << fps_mt << " FPS"
              << "  (加速比 " << fps_mt / fps_single << "x)" << std::endl;
    std::cout << "  每线程吞吐:        " << fps_mt / THREAD_COUNT << " FPS/线程" << std::endl;
    return 0;
}
