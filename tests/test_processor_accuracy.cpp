//
// Created by aichao on 2025/8/2.
// 预处理加速实现（CUDA / SIMD / batch kernel）vs CPU native 准确性对比测试。
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <vector>
#include <cstdint>

#include "csrc/vision.h"
#include "csrc/vision/processors/processor_factory.h"
#include "csrc/vision/processors/cpu/cpu_processor_backend.h"
#include "csrc/vision/processors/cpu/simd/fused_preproc_simd.h"
#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace {
    // 确定性合成图（梯度 + 唯一值，任何错映射都会暴露）
    ImageData make_test_image(int w, int h) {
        std::vector<uint8_t> buf(static_cast<size_t>(w) * h * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int idx = (y * w + x) * 3;
                buf[idx] = static_cast<uint8_t>(x % 256);
                buf[idx + 1] = static_cast<uint8_t>(y % 256);
                buf[idx + 2] = static_cast<uint8_t>((x + y) % 256);
            }
        return ImageData::from_raw(buf.data(), w, h, MdImageType::PKG_BGR_U8, true);
    }

    // fused_preprocess 标量参考实现（与 kernel 同一映射：src=(dst-origin)/scale，越界写 pad(仿射后)）
    void fused_ref(const uint8_t* src, int src_w, int src_h,
                   float* dst, int dst_w, int dst_h,
                   float ox, float oy, float sx, float sy,
                   const float* alpha, const float* beta, bool swap_rb, float pad) {
        const int plane = dst_h * dst_w;
        for (int y = 0; y < dst_h; ++y) {
            const float src_yf = (static_cast<float>(y) - oy) / sy;
            for (int x = 0; x < dst_w; ++x) {
                const float src_xf = (static_cast<float>(x) - ox) / sx;
                const int idx = y * dst_w + x;
                if (src_xf < 0.0f || src_xf >= static_cast<float>(src_w) ||
                    src_yf < 0.0f || src_yf >= static_cast<float>(src_h)) {
                    dst[0 * plane + idx] = pad;
                    dst[1 * plane + idx] = pad;
                    dst[2 * plane + idx] = pad;
                    continue;
                }
                const int sxi = static_cast<int>(src_xf);
                const int syi = static_cast<int>(src_yf);
                const int si = (syi * src_w + sxi) * 3;
                const float b = src[si], g = src[si + 1], r = src[si + 2];
                float c0, c1, c2;
                if (swap_rb) { c0 = r; c1 = g; c2 = b; }
                else { c0 = b; c1 = g; c2 = r; }
                dst[0 * plane + idx] = c0 * alpha[0] + beta[0];
                dst[1 * plane + idx] = c1 * alpha[1] + beta[1];
                dst[2 * plane + idx] = c2 * alpha[2] + beta[2];
            }
        }
    }

    // 对比两个 CPU FP32 tensor，返回最大绝对差与差异像素数
    double tensor_maxdiff(const Tensor& a, const Tensor& b, size_t* ndiff) {
        if (a.byte_size() != b.byte_size()) {
            *ndiff = 999999;
            return 1e9;
        }
        const float* pa = static_cast<const float*>(a.data());
        const float* pb = static_cast<const float*>(b.data());
        const size_t n = a.byte_size() / sizeof(float);
        double md = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = std::fabs(static_cast<double>(pa[i] - pb[i]));
            if (d > md) md = d;
            if (d > 1e-4) ++cnt;
        }
        *ndiff = cnt;
        return md;
    }
} // namespace

// ==================== fused_preprocess：标量 ref vs SIMD vs CUDA ====================
TEST_CASE("Processor accuracy: fused_preprocess scalar vs SIMD", "[processor_accuracy]") {
    const int src_w = 320, src_h = 240;
    auto img = make_test_image(src_w, src_h);
    const std::vector<int> dst{224, 224};
    const float alpha[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    const float beta[3] = {-1.0f, -1.0f, -1.0f};

    // 标量参考
    std::vector<float> ref(3 * 224 * 224);
    fused_ref(img.data(), src_w, src_h, ref.data(), 224, 224,
              0.0f, 0.0f, static_cast<float>(dst[0]) / src_w, static_cast<float>(dst[1]) / src_h,
              alpha, beta, true, 0.0f);

    // SIMD 内核（运行时派发）
    Tensor simd_t;
    simd_t.allocate({3, 224, 224}, DataType::FP32, Device::CPU);
    const auto kernel = get_fused_preproc_kernel();
    kernel(img.data(), src_w, src_h, static_cast<float*>(simd_t.data()), 224, 224,
           0.0f, 0.0f, static_cast<float>(dst[0]) / src_w, static_cast<float>(dst[1]) / src_h,
           alpha, beta, true, 0.0f);
    size_t nd = 0;
    const double md = tensor_maxdiff(simd_t, Tensor(ref.data(), {3, 224, 224}, DataType::FP32, Device::CPU), &nd);
    REQUIRE(md < 1e-5);
    REQUIRE(nd == 0);
}

TEST_CASE("Processor accuracy: fused_preprocess CPU vs CUDA", "[processor_accuracy][gpu]") {
#ifdef WITH_GPU
    const int src_w = 320, src_h = 240;
    auto img = make_test_image(src_w, src_h);
    const std::vector<int> dst{224, 224};
    const float alpha[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    const float beta[3] = {-1.0f, -1.0f, -1.0f};
    const float ox = 0, oy = 0;
    const float sx = static_cast<float>(dst[0]) / src_w, sy = static_cast<float>(dst[1]) / src_h;

    std::vector<float> ref(3 * 224 * 224);
    fused_ref(img.data(), src_w, src_h, ref.data(), 224, 224, ox, oy, sx, sy, alpha, beta, true, 0.0f);

    // CUDA backend
    auto backend = create_processor_backend(Device::GPU, Backend::ORT, 0);
    Tensor cuda_t;
    REQUIRE(backend->fused_preprocess(img, &cuda_t, dst, ox, oy, sx, sy,
                                      std::vector<float>(alpha, alpha + 3),
                                      std::vector<float>(beta, beta + 3), true, 0.0f));
    // CPU backend（SIMD）
    auto cpu_backend = create_processor_backend(Device::CPU, Backend::ORT, 0);
    Tensor cpu_t;
    REQUIRE(cpu_backend->fused_preprocess(img, &cpu_t, dst, ox, oy, sx, sy,
                                          std::vector<float>(alpha, alpha + 3),
                                          std::vector<float>(beta, beta + 3), true, 0.0f));
    // GPU -> host
    std::vector<float> host(cuda_t.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), cuda_t.data(), cuda_t.byte_size(), cudaMemcpyDeviceToHost);
    Tensor cuda_host(host.data(), cuda_t.shape(), DataType::FP32, Device::CPU);
    size_t nd = 0;
    const double md = tensor_maxdiff(cpu_t, cuda_host, &nd);
    REQUIRE(md < 1e-4);
    REQUIRE(nd <= 8); // 允许少数边界像素（不同插值舍入）
#endif
}

// ==================== yolo_preprocess：CPU vs CUDA ====================
TEST_CASE("Processor accuracy: yolo_preprocess CPU vs CUDA", "[processor_accuracy][gpu]") {
#ifdef WITH_GPU
    const int src_w = 480, src_h = 270;
    auto img = make_test_image(src_w, src_h);
    const std::vector<int> dst{640, 640};
    const float pad = 114.0f;

    auto cpu_backend = create_processor_backend(Device::CPU, Backend::ORT, 0);
    auto cuda_backend = create_processor_backend(Device::GPU, Backend::ORT, 0);
    Tensor cpu_t, cuda_t;
    LetterBoxRecord r1, r2;
    REQUIRE(cpu_backend->yolo_preprocess(img, &cpu_t, dst, pad, &r1));
    REQUIRE(cuda_backend->yolo_preprocess(img, &cuda_t, dst, pad, &r2));

    // 校验 letterbox 参数一致
    REQUIRE(r1.scale == Catch::Approx(r2.scale).epsilon(1e-5));
    REQUIRE(r1.pad_w == Catch::Approx(r2.pad_w).epsilon(1e-4));
    REQUIRE(r1.pad_h == Catch::Approx(r2.pad_h).epsilon(1e-4));

    std::vector<float> host(cuda_t.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), cuda_t.data(), cuda_t.byte_size(), cudaMemcpyDeviceToHost);
    Tensor cuda_host(host.data(), cuda_t.shape(), DataType::FP32, Device::CPU);
    size_t nd = 0;
    const double md = tensor_maxdiff(cpu_t, cuda_host, &nd);
    // 调试：找 max-diff 像素
    const float* pa = static_cast<const float*>(cpu_t.data());
    const float* pb = host.data();
    const size_t n = cpu_t.byte_size() / sizeof(float);
    double best = 0; size_t best_i = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)pa[i] - pb[i]);
        if (d > best) { best = d; best_i = i; }
    }
    INFO("maxdiff=" << best << " at index=" << best_i << " cpu=" << pa[best_i] << " cuda=" << pb[best_i]);
    REQUIRE(md < 1e-4);
    REQUIRE(nd <= 16);
#endif
}

// ==================== yolo_preprocess_batch：CPU vs CUDA ====================
TEST_CASE("Processor accuracy: yolo_preprocess_batch CPU vs CUDA", "[processor_accuracy][gpu]") {
#ifdef WITH_GPU
    auto img0 = make_test_image(480, 270);
    auto img1 = make_test_image(640, 360);
    auto img2 = make_test_image(320, 240);
    std::vector<ImageData> imgs = {img0, img1, img2};
    const std::vector<int> dst{640, 640};
    const float pad = 114.0f;

    auto cpu_backend = create_processor_backend(Device::CPU, Backend::ORT, 0);
    auto cuda_backend = create_processor_backend(Device::GPU, Backend::ORT, 0);
    Tensor cpu_t, cuda_t;
    std::vector<LetterBoxRecord> cr, cr2;
    REQUIRE(cpu_backend->yolo_preprocess_batch(imgs, &cpu_t, dst, pad, &cr));
    REQUIRE(cuda_backend->yolo_preprocess_batch(imgs, &cuda_t, dst, pad, &cr2));

    REQUIRE(cpu_t.shape() == cuda_t.shape());
    REQUIRE(cr.size() == cr2.size());
    for (size_t i = 0; i < cr.size(); ++i) {
        REQUIRE(cr[i].scale == Catch::Approx(cr2[i].scale).epsilon(1e-5));
        REQUIRE(cr[i].pad_w == Catch::Approx(cr2[i].pad_w).epsilon(1e-4));
    }

    std::vector<float> host(cuda_t.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), cuda_t.data(), cuda_t.byte_size(), cudaMemcpyDeviceToHost);
    Tensor cuda_host(host.data(), cuda_t.shape(), DataType::FP32, Device::CPU);
    size_t nd = 0;
    const double md = tensor_maxdiff(cpu_t, cuda_host, &nd);
    REQUIRE(md < 1e-4);
    REQUIRE(nd <= 24);
#endif
}

// ==================== scrfd_preprocess：CPU(SIMD) vs CUDA ====================
TEST_CASE("Processor accuracy: scrfd_preprocess CPU vs CUDA", "[processor_accuracy][gpu]") {
#ifdef WITH_GPU
    const int src_w = 256, src_h = 256;
    auto img = make_test_image(src_w, src_h);
    const std::vector<int> dst{248, 248};
    const float pad = 114.0f;

    auto cpu_backend = create_processor_backend(Device::CPU, Backend::ORT, 0);
    auto cuda_backend = create_processor_backend(Device::GPU, Backend::ORT, 0);
    Tensor cpu_t, cuda_t;
    LetterBoxRecord r1, r2;
    REQUIRE(cpu_backend->scrfd_preprocess(img, &cpu_t, dst, pad, &r1));
    REQUIRE(cuda_backend->scrfd_preprocess(img, &cuda_t, dst, pad, &r2));

    std::vector<float> host(cuda_t.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), cuda_t.data(), cuda_t.byte_size(), cudaMemcpyDeviceToHost);
    Tensor cuda_host(host.data(), cuda_t.shape(), DataType::FP32, Device::CPU);
    size_t nd = 0;
    const double md = tensor_maxdiff(cpu_t, cuda_host, &nd);
    REQUIRE(md < 1e-4);
    REQUIRE(nd <= 16);
#endif
}

// ============ fused_preprocess_batch：CPU(SIMD) vs 逐图 与 CUDA(3D grid) ============
TEST_CASE("Processor accuracy: fused_preprocess_batch CPU vs per-image vs CUDA", "[processor_accuracy][gpu]") {
    std::vector<ImageData> imgs = {
        make_test_image(160, 100),
        make_test_image(240, 120),
        make_test_image(96, 160),
    };
    const std::vector<int> dst{224, 224};
    const std::vector<float> alpha = {1.0f / (255.0f * 0.229f), 1.0f / (255.0f * 0.224f), 1.0f / (255.0f * 0.225f)};
    const std::vector<float> beta = {-0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f};

    std::vector<float> oxs(3), oys(3, 0.0f), sxs(3), sys(3);
    for (int i = 0; i < 3; ++i) {
        sxs[i] = static_cast<float>(dst[0]) / imgs[i].width();
        sys[i] = static_cast<float>(dst[1]) / imgs[i].height();
        oxs[i] = 0.0f;
    }

    auto cpu_backend = create_processor_backend(Device::CPU, Backend::ORT, 0);
    Tensor batch_cpu;
    REQUIRE(cpu_backend->fused_preprocess_batch(imgs, &batch_cpu, dst, oxs, oys, sxs, sys,
                                                alpha, beta, true, 0.0f));

    // batch == 逐图 concat（正确性）
    std::vector<Tensor> singles(3);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(cpu_backend->fused_preprocess(imgs[i], &singles[i], dst, oxs[i], oys[i], sxs[i], sys[i],
                                              alpha, beta, true, 0.0f));
    }
    Tensor concat_ref = Tensor::concat(singles, 0);
    size_t nd = 0;
    REQUIRE(tensor_maxdiff(batch_cpu, concat_ref, &nd) < 1e-5);
    REQUIRE(nd == 0);

#ifdef WITH_GPU
    // CPU(SIMD) vs CUDA(3D grid)
    auto cuda_backend = create_processor_backend(Device::GPU, Backend::ORT, 0);
    Tensor batch_cuda;
    REQUIRE(cuda_backend->fused_preprocess_batch(imgs, &batch_cuda, dst, oxs, oys, sxs, sys,
                                                 alpha, beta, true, 0.0f));
    std::vector<float> host(batch_cuda.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), batch_cuda.data(), batch_cuda.byte_size(), cudaMemcpyDeviceToHost);
    Tensor cuda_host(host.data(), batch_cuda.shape(), DataType::FP32, Device::CPU);
    REQUIRE(tensor_maxdiff(batch_cpu, cuda_host, &nd) < 1e-4);
    REQUIRE(nd <= 24);
#endif
}