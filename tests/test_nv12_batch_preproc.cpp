//
// T1: NV12 批量融合预处理（设备零拷贝）正确性测试。
// 每个用例：批(batch kernel, yolo_preprocess_batch) vs 逐帧(yolo_preprocess_nv12) allclose。
// host NV12 帧：kernel 聚合 H2D；device NV12 帧：plane 指针即显存，零 PCIe。
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "csrc/vision.h"
#include "core/tensor.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/processors/processor_factory.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace {

    // 生成确定性的 NV12 图（Y: x+y 渐变，UV: 带图案），返回独立 y/uv 平面缓冲与步长
    struct Nv12 { std::vector<uint8_t> y, uv; int w = 0, h = 0, step_y = 0, step_uv = 0; };

    Nv12 make_host_nv12(int w, int h) {
        Nv12 r;
        r.w = w; r.h = h;
        r.step_y = w; r.step_uv = w;
        r.y.assign(static_cast<size_t>(w) * h, 0);
        r.uv.assign(static_cast<size_t>(w) * (h / 2), 0);
        for (int yy = 0; yy < h; ++yy)
            for (int x = 0; x < w; ++x)
                r.y[yy * w + x] = static_cast<uint8_t>((x + yy) & 0xFF);
        for (size_t i = 0; i < r.uv.size(); ++i)
            r.uv[i] = static_cast<uint8_t>(128 + (i * 7 + (i >> 2)) & 63);
        return r;
    }

    ImageData wrap_host_nv12(Nv12& n) {
        ImageData::Plane pl[2] = {{n.y.data(), n.step_y}, {n.uv.data(), n.step_uv}};
        auto im = ImageData::from_planes(pl, 2, MdImageType::NV12, n.w, n.h, Device::CPU, {});
        return im;
    }

    struct DevNv12 {
        uint8_t* y = nullptr;
        uint8_t* uv = nullptr;
        int w = 0, h = 0, step_y = 0, step_uv = 0;
        DevNv12() = default;
        DevNv12(const DevNv12&) = delete;
        DevNv12& operator=(const DevNv12&) = delete;
        DevNv12(DevNv12&& o) noexcept { *this = std::move(o); }
        DevNv12& operator=(DevNv12&& o) noexcept {
            if (this == &o) return *this;
            if (y) cudaFree(y);
            if (uv) cudaFree(uv);
            y = o.y; uv = o.uv;
            w = o.w; h = o.h; step_y = o.step_y; step_uv = o.step_uv;
            o.y = nullptr; o.uv = nullptr;
            return *this;
        }
        ~DevNv12() {
            if (y) cudaFree(y);
            if (uv) cudaFree(uv);
        }
    };

    DevNv12 to_device(const Nv12& n) {
        DevNv12 d;
        d.w = n.w; d.h = n.h; d.step_y = n.step_y; d.step_uv = n.step_uv;
        cudaMalloc(&d.y, n.y.size());
        cudaMalloc(&d.uv, n.uv.size());
        cudaMemcpy(d.y, n.y.data(), n.y.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d.uv, n.uv.data(), n.uv.size(), cudaMemcpyHostToDevice);
        return d;
    }

    ImageData wrap_device_nv12(DevNv12& d) {
        ImageData::Plane pl[2] = {{d.y, d.step_y}, {d.uv, d.step_uv}};
        auto im = ImageData::from_planes(pl, 2, MdImageType::NV12, d.w, d.h, Device::GPU, {});
        return im;
    }

    double maxdiff(const Tensor& a, const Tensor& b, size_t* ndiff = nullptr) {
        const float* pa = static_cast<const float*>(a.data());
        const float* pb = static_cast<const float*>(b.data());
        const size_t n = a.byte_size() / sizeof(float);
        double best = 0; size_t nd = 0;
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs((double)pa[i] - pb[i]);
            if (d > best) best = d;
            if (d > 1e-5f) ++nd;
        }
        if (ndiff) *ndiff = nd;
        return best;
    }

} // namespace

// host NV12 帧：批 kernel（聚合 H2D） vs 逐帧 yolo_preprocess_nv12
TEST_CASE("NV12 batch preproc: host frames vs per-frame", "[gpu][nv12]") {
#ifdef WITH_GPU
    std::vector<Nv12> srcs;
    srcs.push_back(make_host_nv12(480, 270));
    srcs.push_back(make_host_nv12(640, 360));
    srcs.push_back(make_host_nv12(320, 240));
    const std::vector<int> dst{640, 640};
    const float pad = 114.0f;

    auto cuda_bk = create_processor_backend(Device::GPU, Backend::ORT, 0);
    REQUIRE(cuda_bk != nullptr);

    // 批量：host NV12 → 批 kernel
    std::vector<ImageData> frames;
    for (auto& s : srcs) frames.push_back(wrap_host_nv12(s));
    Tensor batch_t;
    std::vector<LetterBoxRecord> brecs;
    REQUIRE(cuda_bk->yolo_preprocess_batch(frames, &batch_t, dst, pad, &brecs));

    // 逐帧：host NV12 → 单帧 CUDA kernel
    for (size_t b = 0; b < srcs.size(); ++b) {
        Tensor ref_t;
        LetterBoxRecord rrec;
        auto& s = srcs[b];
        REQUIRE(cuda_bk->yolo_preprocess_nv12(s.y.data(), s.uv.data(), {s.w, s.h},
                                              s.step_y, s.step_uv, &ref_t, dst, pad, &rrec,
                                              Device::CPU));

        // 批结果第 b 帧 D2H 到 CPU 后取块
        std::vector<float> host(batch_t.byte_size() / sizeof(float));
        cudaMemcpy(host.data(), batch_t.data(), batch_t.byte_size(), cudaMemcpyDeviceToHost);
        const size_t channels = 3, hh = (size_t)dst[1], ww = (size_t)dst[0], plane = hh * ww;
        const size_t img_stride = channels * plane;
        std::vector<float> slice(channels * plane);
        for (size_t c = 0; c < channels; ++c)
            for (size_t k = 0; k < plane; ++k)
                slice[c * plane + k] = host[b * img_stride + c * plane + k];
        Tensor slice_t(slice.data(), ref_t.shape(), DataType::FP32, Device::CPU);

        std::vector<float> refh(ref_t.byte_size() / sizeof(float));
        cudaMemcpy(refh.data(), ref_t.data(), ref_t.byte_size(), cudaMemcpyDeviceToHost);
        Tensor refh_t(refh.data(), ref_t.shape(), DataType::FP32, Device::CPU);

        size_t nd = 0;
        const double md = maxdiff(slice_t, refh_t, &nd);
        INFO("host frame b=" << b << " maxdiff=" << md << " ndiff=" << nd);
        REQUIRE(md < 1e-4f);
        // letterbox 参数一致
        REQUIRE(brecs[b].scale == Catch::Approx(rrec.scale).epsilon(1e-5));
    }
#else
    REQUIRE(true);
#endif
}

// device NV12 帧：批 kernel 零拷贝 vs 逐帧（设备指针）yolo_preprocess_nv12
TEST_CASE("NV12 batch preproc: device frames vs per-frame", "[gpu][nv12]") {
#ifdef WITH_GPU
    std::vector<DevNv12> devs;
    {
        auto a = make_host_nv12(480, 270);
        auto b = make_host_nv12(640, 360);
        auto c = make_host_nv12(320, 240);
        devs.push_back(to_device(a));
        devs.push_back(to_device(b));
        devs.push_back(to_device(c));
    }
    const std::vector<int> dst{640, 640};
    const float pad = 114.0f;

    auto cuda_bk = create_processor_backend(Device::GPU, Backend::ORT, 0);
    REQUIRE(cuda_bk != nullptr);

    std::vector<ImageData> frames;
    for (auto& d : devs) frames.push_back(wrap_device_nv12(d));
    Tensor batch_t;
    std::vector<LetterBoxRecord> brecs;
    REQUIRE(cuda_bk->yolo_preprocess_batch(frames, &batch_t, dst, pad, &brecs));

    std::vector<float> host(batch_t.byte_size() / sizeof(float));
    cudaMemcpy(host.data(), batch_t.data(), batch_t.byte_size(), cudaMemcpyDeviceToHost);

    for (size_t b = 0; b < devs.size(); ++b) {
        Tensor ref_t;
        LetterBoxRecord rrec;
        auto& d = devs[b];
        REQUIRE(cuda_bk->yolo_preprocess_nv12(d.y, d.uv, {d.w, d.h}, d.step_y, d.step_uv,
                                              &ref_t, dst, pad, &rrec, Device::GPU));

        const size_t channels = 3, hh = (size_t)dst[1], ww = (size_t)dst[0], plane = hh * ww;
        const size_t img_stride = channels * plane;
        std::vector<float> slice(channels * plane);
        for (size_t c = 0; c < channels; ++c)
            for (size_t k = 0; k < plane; ++k)
                slice[c * plane + k] = host[b * img_stride + c * plane + k];
        Tensor slice_t(slice.data(), ref_t.shape(), DataType::FP32, Device::CPU);

        std::vector<float> refh(ref_t.byte_size() / sizeof(float));
        cudaMemcpy(refh.data(), ref_t.data(), ref_t.byte_size(), cudaMemcpyDeviceToHost);
        Tensor refh_t(refh.data(), ref_t.shape(), DataType::FP32, Device::CPU);

        size_t nd = 0;
        const double md = maxdiff(slice_t, refh_t, &nd);
        INFO("device frame b=" << b << " maxdiff=" << md << " ndiff=" << nd);
        REQUIRE(md < 1e-4f);
        REQUIRE(brecs[b].scale == Catch::Approx(rrec.scale).epsilon(1e-5));
    }
#else
    REQUIRE(true);
#endif
}
