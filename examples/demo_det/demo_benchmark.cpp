//
// 纯推理吞吐测试（排除预处理/结果转换开销）
//
#include "csrc/vision.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace modeldeploy::vision;

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(4);
    option.use_ort_backend();
    option.use_gpu(0);
    option.enable_fp16 = true;
    option.enable_trt = false;
    option.ort_option.trt_engine_cache_path = "./trt_engine";

    auto det = modeldeploy::vision::detection::UltralyticsDet(
        "../../test_data/test_models/yolo11n_nms.onnx", option);
    det.get_preprocessor().use_cuda_preproc();
    det.get_preprocessor().set_size({640, 640});

    // 使用多张不同图片（避免缓存）
    std::vector<ImageData> images;
    for (auto& p : {"test_detection0.jpg", "111.jpg", "best_0.jpg", "2341.jpg"}) {
        auto img = ImageData::imread(std::string("../../test_data/test_images/") + p);
        if (!img.empty()) images.push_back(img);
    }
    if (images.size() < 2) { std::cerr << "need images" << std::endl; return 1; }

    // 预热 + 跳过 ORT 初始化
    std::vector<DetectionResult> result;
    for (int i = 0; i < 30; ++i) det.predict(images[i % images.size()], &result);

    // 测试 1: 单线程纯推理
    const int N = 500;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) det.predict(images[i % images.size()], &result);
    auto t1 = std::chrono::steady_clock::now();
    auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double fps1 = 1000.0 * N / ms1;

    std::cout << "\n单线程 " << N << " 帧: "
              << ms1 << "ms  "
              << fps1 << " FPS  "
              << (double)ms1 / N << "ms/帧" << std::endl;

    // 测试 2: 4 clone 并发
    std::vector<std::unique_ptr<decltype(det)>> clones;
    for (int t = 0; t < 4; ++t) clones.push_back(det.clone());

    auto t2 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&clones, t, &images, N]() {
            std::vector<DetectionResult> r;
            for (int i = 0; i < N; ++i) clones[t]->predict(images[(t + i) % images.size()], &r);
        });
    }
    for (auto& th : threads) th.join();
    auto t3 = std::chrono::steady_clock::now();
    auto ms4 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    double fps4 = 1000.0 * N * 4 / ms4;

    std::cout << "4 clone " << (N * 4) << " 帧: "
              << ms4 << "ms  "
              << fps4 << " FPS  "
              << (double)ms4 / N / 4 << "ms/帧(路)"
              << "  加速比 " << fps4 / fps1 << "x" << std::endl;

    return 0;
}
