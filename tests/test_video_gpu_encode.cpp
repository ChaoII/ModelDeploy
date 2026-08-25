#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_decoder.h"
#include "csrc/video/video_frame.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/ffmpeg_encoder.h"
#include "csrc/video/backend/gst_encoder.h"
#include <algorithm>
#include <cstring>
#include <memory>

// GPU 直接编码 encode_from_gpu_nv12：设备 NV12 指针(Y 平面 d_y / UV 平面 d_uv)直编，省主机往返。
// 本用例 [video][gpu]：需要真实 CUDA 设备 + nvenc/nvh264enc，能力不具备则运行时 SKIP。
// 设备源用 CUDA 运行时 cudaMalloc 分配紧凑连续 NV12（pitch==width，d_uv=d_y+w*h），
// 与上层实际产源（batch 推理/流编码的 GPU NV12）一致。仅当 CMake 提供 CUDA 测试宏
// MODELDEPLOY_TEST_CUDA（GSTREAMER_HAS_CUDA 或 WITH_GPU）时设备源可用，否则整体 SKIP。

using namespace modeldeploy::video;

#ifdef MODELDEPLOY_TEST_CUDA
#include <cuda_runtime.h>
#endif

namespace {

#ifdef MODELDEPLOY_TEST_CUDA
// cudaMalloc 分配一块紧凑连续设备 NV12（W×H → 总字节 w*h*3/2，UV 紧随 Y）。
bool alloc_gpu_nv12(int w, int h, uint8_t*& y, uint8_t*& uv, void*& dev) {
    size_t nbytes = (size_t)w * h * 3 / 2;
    void* p = nullptr;
    if (cudaMalloc(&p, nbytes) != cudaSuccess || !p) return false;
    // 填 128 灰（Y=UV=128，近似中性灰），保证编码/解码正常
    cudaMemset(p, 128, nbytes);
    cudaDeviceSynchronize();
    y = (uint8_t*)p;
    uv = y + (size_t)w * h;
    dev = p;
    return true;
}
#endif

bool has_gpu_nvenc() {
    auto cap = query_video_capabilities();
    return std::find(cap.hw_encoders.begin(), cap.hw_encoders.end(), "h264_nvenc") !=
           cap.hw_encoders.end();
}

} // namespace

#ifdef MODELDEPLOY_TEST_CUDA
// ── FFmpeg：h264_nvenc 吃设备 NV12 指针直编 → soft 回读 ──────────────────────
TEST_CASE("FFmpeg encode_from_gpu_nv12 设备指针直编 nvenc → soft 回读",
          "[video][gpu][ffmpeg][integration]") {
    if (!has_gpu_nvenc()) SKIP("no h264_nvenc in this environment");
    const int W = 192, H = 144;  // 高于 NVENC 最小宽度（146）

    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    void* dev = nullptr;
    if (!alloc_gpu_nv12(W, H, d_y, d_uv, dev)) SKIP("cudaMalloc device NV12 unavailable");
    REQUIRE(d_y != nullptr);
    REQUIRE(d_uv != nullptr);

    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::FFmpeg;
    ecfg.hw_accel = HwAccel::Cuda;
    ecfg.gpu_direct_input = true;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = create_encoder_backend(ecfg);
    REQUIRE(enc != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegEncoder>(enc);
    std::string err;
    bool ok = enc->open("test_data/video/gpu_direct_ffmpeg.mp4", W, H, 25, ecfg, &err);
    REQUIRE(ok);
    REQUIRE(ff->used_hw());  // 确实决议出 nvenc 硬编直编

    modeldeploy::vision::ImageData::Plane planes[2] = {{d_y, W}, {d_uv, W}};
    auto dev_owner = std::shared_ptr<void>(dev, [](void* p) { cudaFree(p); });
    modeldeploy::vision::ImageData gpu_nv12 = modeldeploy::vision::ImageData::from_planes(
        planes, 2, MdImageType::NV12, W, H, modeldeploy::Device::GPU, dev_owner);
    REQUIRE(gpu_nv12.device() == modeldeploy::Device::GPU);

    // 设备 NV12 平面（device=GPU）直编，不做主机往返
    for (int i = 0; i < 24; ++i) {
        REQUIRE(enc->encode(VideoFrame{gpu_nv12}, &err));
    }
    enc->close();

    // soft 软解回读：产物必须是可解码 h264
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec->open("test_data/video/gpu_direct_ffmpeg.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}

// ── GStreamer：nvh264enc 吃 CUDA memory（包装设备 NV12 指针）直编 → soft 回读 ────
#if defined(ENABLE_GSTREAMER) && defined(HAVE_GSTCUDA)
TEST_CASE("GStreamer encode_from_gpu_nv12 CUDA memory 直编 nvh264enc → soft 回读",
          "[video][gpu][gst][gst-cuda][integration]") {
    if (!has_gpu_nvenc()) SKIP("no nvenc in this environment");
    const int W = 192, H = 144;

    uint8_t* d_y = nullptr;
    uint8_t* d_uv = nullptr;
    void* dev = nullptr;
    if (!alloc_gpu_nv12(W, H, d_y, d_uv, dev)) SKIP("cudaMalloc device NV12 unavailable");

    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::GStreamer;
    ecfg.hw_accel = HwAccel::Cuda;
    ecfg.gpu_direct_input = true;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = create_encoder_backend(ecfg);
    REQUIRE(enc != nullptr);
    std::string err;
    bool ok = enc->open("test_data/video/gpu_direct_gst.mp4", W, H, 25, ecfg, &err);
    if (!ok) {
        // 设备 CUDA 上下文/插件不可用属环境能力不足 → SKIP（如实探测，非 FAIL）
        if (err.rfind("cuda-", 0) == 0 || err.rfind("nvcodec-", 0) == 0 ||
            err.rfind("no-nvh264enc", 0) == 0 || err.rfind("gpu-direct-needs", 0) == 0)
            SKIP("GStreamer CUDA encode path unavailable: " + err);
        FAIL("gpu_direct gst open failed: " + err);
    }
    auto gst = std::dynamic_pointer_cast<GstEncoder>(enc);
    REQUIRE(gst->used_hw());

    modeldeploy::vision::ImageData::Plane planes[2] = {{d_y, W}, {d_uv, W}};
    auto dev_owner = std::shared_ptr<void>(dev, [](void* p) { cudaFree(p); });
    modeldeploy::vision::ImageData gpu_nv12 = modeldeploy::vision::ImageData::from_planes(
        planes, 2, MdImageType::NV12, W, H, modeldeploy::Device::GPU, dev_owner);
    REQUIRE(gpu_nv12.device() == modeldeploy::Device::GPU);

    for (int i = 0; i < 24; ++i) {
        REQUIRE(enc->encode(VideoFrame{gpu_nv12}, &err));
    }
    enc->close();

    // soft 软解回读验证产物 h264
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec->open("test_data/video/gpu_direct_gst.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}
#endif
#endif
