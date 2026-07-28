//
// Created by aichao on 2025/2/24.
// 单线程 vs 多线程（clone）推理性能对比
// 模拟多路视频流场景：每路处理自己的图像，统计总吞吐
//

#include "csrc/vision.h"
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <sstream>

using namespace modeldeploy::vision;

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(4);
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

    // 读取多张不同图像（模拟多路摄像头）
    std::vector<std::string> img_paths = {
        "../../test_data/test_images/test_detection0.jpg",
        "../../test_data/test_images/111.jpg",
        "../../test_data/test_images/best_0.jpg",
        "../../test_data/test_images/2341.jpg",
    };

    std::vector<ImageData> images;
    for (const auto& p : img_paths) {
        auto img = ImageData::imread(p);
        if (!img.empty()) images.push_back(std::move(img));
    }
    if (images.size() < 2) {
        std::cerr << "需要至少 2 张测试图片" << std::endl;
        return 1;
    }
    int num_images = static_cast<int>(images.size());
    std::cout << "使用 " << num_images << " 张图片，每路反复使用" << std::endl;

    // 预热
    std::vector<DetectionResult> warmup_result;
    for (int i = 0; i < 10; ++i) {
        yolo11_det.predict(images[i % num_images], &warmup_result);
    }
    std::cout << "Warm-up done." << std::endl;

    constexpr int TOTAL_FRAMES = 400;
    constexpr int THREAD_COUNT = 4;

    // ════════════════════════════════════════════════════════
    // 单线程：顺序处理 TOTAL_FRAMES 帧
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 单线程 =====" << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < TOTAL_FRAMES; ++i) {
        yolo11_det.predict(images[i % num_images], &warmup_result);
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms_single = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double fps_single = 1000.0 * TOTAL_FRAMES / ms_single;
    std::cout << "  处理 " << TOTAL_FRAMES << " 帧"
              << "  耗时 " << ms_single << "ms"
              << "  平均 " << (double)ms_single / TOTAL_FRAMES << "ms/帧"
              << "  " << fps_single << " FPS" << std::endl;

    // ════════════════════════════════════════════════════════
    // 多线程：每路 clone 一份，各自处理自己的帧
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 多线程 (" << THREAD_COUNT << " 路, clone) =====" << std::endl;

    std::vector<std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet>> clones;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        clones.push_back(yolo11_det.clone());
    }

    int frames_per_thread = TOTAL_FRAMES / THREAD_COUNT;
    auto t2 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        threads.emplace_back([&clones, t, &images, num_images, frames_per_thread]() {
            std::vector<DetectionResult> r;
            for (int i = 0; i < frames_per_thread; ++i) {
                clones[t]->predict(images[(t + i) % num_images], &r);
            }
        });
    }
    for (auto& th : threads) th.join();

    auto t3 = std::chrono::steady_clock::now();
    auto ms_mt = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    double fps_mt = 1000.0 * TOTAL_FRAMES / ms_mt;
    double avg_per_stream = 1000.0 * frames_per_thread / ms_mt;

    std::cout << "  处理 " << TOTAL_FRAMES << " 帧 (" << THREAD_COUNT
              << " 路 × " << frames_per_thread << " 帧)"
              << "  耗时 " << ms_mt << "ms" << std::endl;
    std::cout << "  " << fps_mt << " FPS (合计)"
              << "  每路 " << avg_per_stream << " FPS/路"
              << "  加速比 " << fps_mt / fps_single << "x" << std::endl;

    // ════════════════════════════════════════════════════════
    // 对比总结
    // ════════════════════════════════════════════════════════
    std::cout << "\n===== 对比 =====" << std::endl;
    std::cout << "  单线程(" << TOTAL_FRAMES << "帧): " << fps_single << " FPS" << std::endl;
    std::cout << "  多线程" << THREAD_COUNT << "路(clone): " << fps_mt << " FPS"
              << "  (加速比 " << fps_mt / fps_single << "x)" << std::endl;

    if (fps_mt > fps_single * 1.1) {
        std::cout << "  ✅ 多线程有显著加速——CPU 预处理是瓶颈，多线程隐藏了延迟" << std::endl;
    } else {
        std::cout << "  ⚠️ GPU 已是瓶颈，单线程已喂饱 GPU。多线程的价值不在单 GPU 加速，" << std::endl;
        std::cout << "     而在同时处理多路视频流（如 20 路 RTSP 各跑 25 FPS）。" << std::endl;
    }
    return 0;
}
