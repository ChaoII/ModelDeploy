#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/ffmpeg_decoder.h"
#include "csrc/video/backend/gst_decoder.h"
#include <fstream>
#include <algorithm>

using namespace modeldeploy::video;
using modeldeploy::Device;

// 设备直通解码（device_only=true）：解码输出必须保持 GPU 设备帧（IPlaneView.device=Device::GPU），
// 且 owner 保活设备的 hw frame/sample，不做 D2H 主机拷贝。
// 本用例 [video][gpu]：需要真实 CUDA 设备（RTX）才能实跑；设备/后端不可用时 SKIP。

namespace {
// 检查本地测试片是否存在
bool has_clip() {
    std::ifstream p("test_data/video/clip.h264");
    return p.good();
}
} // namespace

// ── FFmpeg ─────────────────────────────────────────────────────────────────

TEST_CASE("FFmpeg device_only=true 解码输出 GPU 设备帧 (device==GPU)", "[video][gpu][ffmpeg][integration]") {
    if (!has_clip()) SKIP("no test clip; place at test_data/video/clip.h264");

    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Cuda;
    cfg.device_only = true;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);

    std::string err;
    if (!d->open("test_data/video/clip.h264", &err)) {
        // 设备直通路径在无 CUDA 硬件/解码器时不可用 → 能力未就绪则 SKIP（如实探测）；
        // 若以 “cuda-” 前缀明确报告设备不可用，属环境缺硬件，SKIP 而非 FAIL。
        if (err.rfind("cuda-", 0) == 0) SKIP("CUDA device path unavailable: " + err);
        // 其余错误是真实失败
        FAIL("device_only open failed: " + err);
    }
    REQUIRE(d->width() > 0);
    REQUIRE(d->height() > 0);
    if (ff) INFO("used_hw=" << ff->used_hw());

    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == d->width());
        REQUIRE(f.image.height() == d->height());
        // 核心断言：设备直通解码输出是 GPU 设备帧，不是 CPU 主机帧
        REQUIRE(f.image.device() == Device::GPU);
        // 平面非空（Y/UV 设备指针）
        REQUIRE(f.image.plane_count() == 2);
        REQUIRE(f.image.plane(0).data != nullptr);
        REQUIRE(f.image.plane(1).data != nullptr);
        ++n;
    }
    REQUIRE(n > 0);   // 设备直通必须解出真实帧
    if (ff) REQUIRE(ff->used_hw());  // 确实走了 CUDA 硬解
    d->close();
}

TEST_CASE("FFmpeg device_only=false 输出仍为 CPU 帧 (device==CPU)", "[video][gpu][ffmpeg][integration]") {
    if (!has_clip()) SKIP("no test clip; place at test_data/video/clip.h264");

    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Cuda;
    cfg.device_only = false;  // 默认：硬解仍回 CPU NV12（现状维持）
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    std::string err;
    if (!d->open("test_data/video/clip.h264", &err)) {
        // 无设备时可回退软解也应能 open；若连软解都失败则属环境问题→SKIP
        if (err.rfind("cuda-", 0) == 0) SKIP("CUDA path unavailable: " + err);
        FAIL("open failed: " + err);
    }
    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.device() == Device::CPU);  // 非 device_only 必须是 CPU 帧
        ++n;
    }
    REQUIRE(n > 0);
    d->close();
}

// ── GStreamer ──────────────────────────────────────────────────────────────

TEST_CASE("GStreamer device_only=true 解码输出 GPU 设备帧 (device==GPU)", "[video][gpu][gst][gst-cuda][integration]") {
    if (!has_clip()) SKIP("no test clip; place at test_data/video/clip.h264");

    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Cuda;
    cfg.device_only = true;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    auto gst = std::dynamic_pointer_cast<GstDecoder>(d);

    std::string err;
    if (!d->open("test_data/video/clip.h264", &err)) {
        if (err.rfind("nvcodec-", 0) == 0 || err.rfind("cuda-", 0) == 0)
            SKIP("GStreamer CUDA device path unavailable: " + err);
        FAIL("device_only open failed: " + err);
    }
    REQUIRE(d->width() > 0);
    REQUIRE(d->height() > 0);

    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.device() == Device::GPU);
        REQUIRE(f.image.plane_count() == 2);
        REQUIRE(f.image.plane(0).data != nullptr);
        REQUIRE(f.image.plane(1).data != nullptr);
        ++n;
    }
    REQUIRE(n > 0);
    d->close();
}

TEST_CASE("GStreamer device_only=false 输出仍为 CPU 帧 (device==CPU)", "[video][gpu][gst][gst-cuda][integration]") {
    if (!has_clip()) SKIP("no test clip; place at test_data/video/clip.h264");

    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Auto;
    cfg.device_only = false;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    std::string err;
    if (!d->open("test_data/video/clip.h264", &err)) {
        if (err.rfind("cuda-", 0) == 0) SKIP("CUDA path unavailable: " + err);
        FAIL("open failed: " + err);
    }
    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.device() == Device::CPU);
        ++n;
    }
    REQUIRE(n > 0);
    d->close();
}
