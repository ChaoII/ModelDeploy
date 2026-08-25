#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_decoder.h"
#include "csrc/video/video_encoder.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/backend/encoder_backend.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"
#include <fstream>
#include <string>

// VAAPI / Sophgo 硬件帧路径（W3）。
//
// 本机（Windows）无 VAAPI（需 Linux+libva）与 Sophgo（需 TPU+sophon-mw），因此这里只验证
// **本机可测**的「配置接受 / codec 选择 / 显式请求 fail-closed」决策逻辑；真实 VAAPI/Sophgo 帧
// 路径在对应平台（[vaapi]/[sophgo] tag，能力就绪才实跑，否则 SKIP 或不注册）。
// 绝不在本机宣称 VAAPI/Sophgo 实跑通过。

using namespace modeldeploy::video;

namespace {
bool has_clip() {
    std::ifstream p("test_data/video/clip.h264");
    return p.good();
}
} // namespace

// ── 配置/枚举接受（后端无关，本机可测）─────────────────────────────────────
TEST_CASE("HwAccel Vaapi/Sophgo 枚举映射与 config 校验接受", "[video][hwpath][core]") {
    REQUIRE(hwaccel_to_string(HwAccel::Vaapi) == "vaapi");
    REQUIRE(hwaccel_to_string(HwAccel::Sophgo) == "sophgo");
    std::string err;

    VideoDecoderConfig d;
    d.hw_accel = HwAccel::Vaapi;
    REQUIRE(d.validate(&err));
    d.hw_accel = HwAccel::Sophgo;
    REQUIRE(d.validate(&err));

    VideoEncoderConfig e;
    e.hw_accel = HwAccel::Vaapi;
    e.set_codec("auto").set_format("mp4");
    e.fps = 25;
    e.set_bitrate_kbps(800);
    REQUIRE(e.validate(&err));
    e.hw_accel = HwAccel::Sophgo;
    REQUIRE(e.validate(&err));
}

// ── Sophgo 解码：video 模块未接线（需 sophon-mw），显式请求 fail-closed ────
TEST_CASE("FFmpeg 显式 Sophgo 解码 fail-closed", "[video][hwpath]") {
    if (!has_clip()) SKIP("no test clip; place at test_data/video/clip.h264");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Sophgo;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    std::string err;
    // 必须失败且报 sophgo-requires，绝不静默回退软解
    REQUIRE_FALSE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(err == "sophgo-decode-requires-sophonmw");
}

#ifdef ENABLE_GSTREAMER
TEST_CASE("GStreamer 显式 Sophgo 解码 fail-closed", "[video][hwpath][gst]") {
    if (!has_clip()) SKIP("no test clip");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Sophgo;
    auto d = create_decoder_backend(cfg);
    if (!d) SKIP("GStreamer backend unavailable");
    std::string err;
    REQUIRE_FALSE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(err == "sophgo-decode-requires-sophonmw");
}
#endif

// ── 显式 Vaapi 解码：未编译 VAAPI 时 fail-closed（本机断言）────────────────
#ifndef ENABLE_VAAPI
TEST_CASE("FFmpeg 显式 Vaapi 解码 fail-closed（未编译 VAAPI）", "[video][hwpath]") {
    if (!has_clip()) SKIP("no test clip");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Vaapi;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    std::string err;
    REQUIRE_FALSE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(err == "vaapi-unavailable");
}

#ifdef ENABLE_GSTREAMER
TEST_CASE("GStreamer 显式 Vaapi 解码 fail-closed（未编译 VAAPI）", "[video][hwpath][gst]") {
    if (!has_clip()) SKIP("no test clip");
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::GStreamer;
    cfg.hw_accel = HwAccel::Vaapi;
    auto d = create_decoder_backend(cfg);
    if (!d) SKIP("GStreamer backend unavailable");
    std::string err;
    REQUIRE_FALSE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(err == "vaapi-unavailable");
}
#endif
#endif // !ENABLE_VAAPI

// ── FFmpeg 编码器 codec 选择：vaapih264enc/h264_vaapi ─────────────────────
// 未编译 VAAPI 时映射不到真实编码器 → fail-closed "unsupported-codec"（本机断言）。
#ifndef ENABLE_VAAPI
TEST_CASE("FFmpeg codec=vaapih264enc/h264_vaapi 未编译 VAAPI 时 fail-closed",
          "[video][hwpath]") {
    VideoEncoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Vaapi;
    cfg.fps = 25;
    cfg.set_bitrate_kbps(800);
    cfg.set_format("mp4");
    auto enc = create_encoder_backend(cfg);
    REQUIRE(enc != nullptr);
    for (const char* codec : {"vaapih264enc", "h264_vaapi"}) {
        cfg.set_codec(codec);
        std::string err;
        bool ok = enc->open("hwpath_vaapi_test.mp4", 320, 240, 25, cfg, &err);
        REQUIRE_FALSE(ok);
        REQUIRE(err == "unsupported-codec");
    }
    enc->close();
}
#endif // !ENABLE_VAAPI

// ── 真实 VAAPI / Sophgo 硬件路径（仅对应平台编译/实跑，本机不注册）────────
// 这些用例在 ENABLE_VAAPI（Linux+libva）构建下才存在，且需真实 VAAPI 设备才实跑，
// 否则运行时 SKIP。Sophgo 真实帧路径未实现（需 sophon-mw 集成），故始终 SKIP + 标注。
#ifdef ENABLE_VAAPI
TEST_CASE("FFmpeg VAAPI 硬解 → CPU NV12（仅 Linux VAAPI 实跑）", "[video][vaapi][hwpath][gpu]") {
    // 本机（Windows，无 ENABLE_VAAPI/libva）不会进入此分支；Linux VAAPI 环境无设备时 SKIP。
    SKIP("需要真实 Linux VAAPI 设备；本机构建未编译 ENABLE_VAAPI，未在本机验证");
}
TEST_CASE("FFmpeg VAAPI 硬编 h264_vaapi（仅 Linux VAAPI 实跑）", "[video][vaapi][hwpath][gpu]") {
    SKIP("需要真实 Linux VAAPI 设备；未在本机验证");
}
#endif

// Sophgo 真实解码路径：video 模块未接线（需 sophon-mw bm_video_decode），无论是否
// ENABLE_SOPHGO 都 fail-closed 于 "sophgo-decode-requires-sophonmw"（见上方用法例断言）。
