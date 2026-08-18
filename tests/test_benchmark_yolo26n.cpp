//
// yolo26n_b8.engine（TRT 动态批）吞吐基准：measure batch_predict FPS at batch=1/4/8。
// 目标：20 路 × 25fps = 总 ≥500fps（聚合成帧率 = FPS_total；每路帧率 = FPS_total/20）。
// 运行: ./test_modeldeploy "yolo26n TRT batch throughput"
//
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

#include "csrc/vision.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace fs = std::filesystem;

TEST_CASE("yolo26n TRT batch throughput", "[benchmark][gpu][trt]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    const fs::path root = env ? fs::path(env) / "test_data" : fs::path("test_data");
    const fs::path eng = root / "test_models/trt/yolo26n_b8.engine";
    if (!fs::exists(eng)) { std::cout << "skip: no yolo26n_b8.engine" << std::endl; return; }

    // 收集最多 12 张 jpg 作为 batch 素材
    const fs::path img_dir = root / "test_images";
    std::vector<std::string> files;
    for (auto& e : fs::directory_iterator(img_dir)) {
        if (files.size() >= 12) break;
        auto ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::cout << "skip: no test images" << std::endl; return; }

    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    detection::UltralyticsDet model(eng.string(), opt);
    if (!model.is_initialized()) { std::cout << "skip: TRT engine failed to init" << std::endl; return; }

    // 加载图片（最多 8 张）
    std::vector<ImageData> imgs;
    for (size_t i = 0; i < files.size() && i < 8; ++i) {
        auto im = ImageData::imread(files[i]);
        if (!im.empty()) imgs.push_back(std::move(im));
    }
    if (imgs.empty()) { std::cout << "skip: no loadable images" << std::endl; return; }
    std::cout << "[bm] images=" << imgs.size() << " model=" << eng.string() << std::endl;

    const int batches[] = {1, 4, 8};
    for (int B : batches) {
        if (B > (int)imgs.size()) break;
        std::vector<ImageData> batch(imgs.begin(), imgs.begin() + B);

        // 预热：分配设备缓冲 + kernel 调优
        std::vector<std::vector<DetectionResult>> r;
        for (int i = 0; i < 20; ++i) { model.batch_predict(batch, &r, nullptr); }

        // 计时
        const int M = (B == 1) ? 800 : ((B == 4) ? 400 : 200);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < M; ++i) { model.batch_predict(batch, &r, nullptr); }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double fps = 1000.0 * (M * B) / ms;
        const double ms_frame = ms / (M * B);

        std::cout << "[bm] batch=" << B << ": " << (M * B) << " frames in "
                  << ms << "ms → " << fps << " fps  (" << ms_frame
                  << "ms/frame)  → 可承载约 " << (int)(fps / 25)
                  << " 路×25fps  /  20 路时每路 " << (fps / 20) << "fps" << std::endl;
    }
    REQUIRE(true);
}

// 720p（1280x720）NV12 源的 batch 吞吐：判断 app 3ms/帧是否是 720p 源的真实内核上限
TEST_CASE("yolo26n TRT 720p NV12 throughput", "[benchmark][gpu][trt]") {
    const char* env = std::getenv("TEST_DATA_DIR");
    const fs::path root = env ? fs::path(env) / "test_data" : fs::path("test_data");
    const fs::path eng = root / "test_models/trt/yolo26n_b8.engine";
    if (!fs::exists(eng)) return;

    RuntimeOption opt;
    opt.use_gpu(0);
    opt.use_trt_backend();
    detection::UltralyticsDet model(eng.string(), opt);
    if (!model.is_initialized()) return;

    // 合成 1280x720 host NV12
    const int w = 1280, h = 720;
    std::vector<uint8_t> y((size_t)w * h), uv((size_t)w * h / 2);
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            y[(size_t)i * w + j] = (uint8_t)((i + j) & 0xFF);
    for (size_t i = 0; i < uv.size(); ++i) uv[i] = (uint8_t)(128 + (i * 7 & 63));

    auto make_nv12 = [&](int idx) {
        ImageData::Plane pl[2] = {{y.data(), w}, {uv.data(), w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, Device::CPU, {});
    };

    // 设备 NV12（零拷贝 fast path 对照）：cudaMalloc 紧凑 + D2H 复制，owner=delereter cudaFree
    struct CudaDeleter {
        void operator()(void* p) const { if (p) cudaFree(p); }
    };
    std::shared_ptr<uint8_t> dblk;
    uint8_t* dbuf = nullptr;
    cudaMalloc(&dbuf, (size_t)w * h * 3 / 2);
    dblk = std::shared_ptr<uint8_t>(dbuf, CudaDeleter{});
    {
        cudaMemcpy(dbuf, y.data(), (size_t)w * h, cudaMemcpyHostToDevice);
        cudaMemcpy(dbuf + (size_t)w * h, uv.data(), (size_t)w * h / 2,
                   cudaMemcpyHostToDevice);
    }
    auto make_nv12_dev = [&](int idx) {
        ImageData::Plane pl[2] = {{dbuf, w}, {dbuf + (size_t)w * h, w}};
        return ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, Device::GPU, dblk);
    };

    const int batches[] = {4, 8};
    for (int B : batches) {
        std::vector<ImageData> imgs;
        for (int i = 0; i < B; ++i) imgs.push_back(make_nv12(i));
        std::vector<std::vector<DetectionResult>> r;
        for (int i = 0; i < 10; ++i) model.batch_predict(imgs, &r, nullptr);
        const int M = 150;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < M; ++i) model.batch_predict(imgs, &r, nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double fps = 1000.0 * (M * B) / ms;
        std::cout << "[bm720p-host] batch=" << B << ": " << fps << " fps  ("
                  << ms / (M * B) << " ms/frame)  det=" << r[0].size() << std::endl;
    }
    for (int B : batches) {
        std::vector<ImageData> imgs;
        for (int i = 0; i < B; ++i) imgs.push_back(make_nv12_dev(i));
        std::vector<std::vector<DetectionResult>> r;
        for (int i = 0; i < 10; ++i) model.batch_predict(imgs, &r, nullptr);
        const int M = 150;
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < M; ++i) model.batch_predict(imgs, &r, nullptr);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double fps = 1000.0 * (M * B) / ms;
        std::cout << "[bm720p-dev] batch=" << B << ": " << fps << " fps  ("
                  << ms / (M * B) << " ms/frame)  det=" << r[0].size() << std::endl;
    }
    REQUIRE(true);
}

