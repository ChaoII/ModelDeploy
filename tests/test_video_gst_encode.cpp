#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/gst_encoder.h"
#include <algorithm>
#include <cstring>

using namespace modeldeploy::video;

TEST_CASE("GStreamer x264enc 编码 mp4 → 回读验证", "[video][gst][integration]") {
    auto cap = query_video_capabilities();
    if (!cap.gstreamer_available) SKIP("no GStreamer runtime");
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::GStreamer;
    ecfg.set_codec("x264enc").set_format("mp4");
    auto enc = VideoEncoder::create(ecfg);
    if (!enc) SKIP("no x264enc plugin");  // 无 x264 插件环境跳过
    std::string err;
    REQUIRE(enc->open("test_data/video/gst_out.mp4", 64, 64, 25, &err));
    // 生成 32 帧纯色 BGR ImageData 逐帧 encode
    uint8_t bgr[64 * 64 * 3];
    memset(bgr, 128, sizeof(bgr));
    auto img = modeldeploy::vision::ImageData::from_raw(bgr, 64, 64, MdImageType::PKG_BGR_U8,
                                                        true);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    // 用 Task5 GStreamer 解码器/FFmpeg 解码器回读 gst_out.mp4 验证帧数
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;  // 本用例验证 GStreamer 编码产物可解码，固定软解（低分辨率 cuvid 可能不支持）
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec != nullptr);
    REQUIRE(dec->open("test_data/video/gst_out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
    REQUIRE(cnt <= 32 + 5);  // 容忍首帧延迟误差
}

// nvh264enc 硬编：能力具备才跑（nvh264enc 在 hw_encoders），软解回读>0 帧。
// 本卡 NVENC 最小宽=146（H3 实测），用 192×144 稳定稳妥。
TEST_CASE("GStreamer nvh264enc 硬编 → mp4 回读", "[video][hw][gpu][integration]") {
    auto cap = query_video_capabilities();
    bool nv = std::find(cap.hw_encoders.begin(), cap.hw_encoders.end(), "nvh264enc") !=
              cap.hw_encoders.end();
    if (!nv) SKIP("no nvh264enc encoder in this environment");
    const int W = 192, H = 144;
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::GStreamer;
    ecfg.hw_accel = HwAccel::Cuda;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = create_encoder_backend(ecfg);
    if (!enc) SKIP("no GStreamer encoder");  // runtime_available 需 x264enc
    auto gst = std::dynamic_pointer_cast<GstEncoder>(enc);
    std::string err;
    REQUIRE(enc->open("test_data/video/gst_nv_out.mp4", W, H, 25, ecfg, &err));
    REQUIRE(gst != nullptr);
    REQUIRE(gst->used_hw());  // auto+Cuda 且 nvh264enc 存在 → 必须真正走 nv 硬编，而非静默回退 x264
    uint8_t bgr[W * H * 3];
    memset(bgr, 128, sizeof(bgr));
    auto img = modeldeploy::vision::ImageData::from_raw(bgr, W, H, MdImageType::PKG_BGR_U8, true);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    // 软解回读验证产物可解码
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec != nullptr);
    REQUIRE(dec->open("test_data/video/gst_nv_out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}

TEST_CASE("GStreamer qsvh264enc 硬编（CPU BGR→QSV）→ mp4 回读", "[video][gst][hw][gpu][integration]") {
    auto cap = query_video_capabilities();
    bool qsv = std::find(cap.hw_encoders.begin(), cap.hw_encoders.end(), "qsvh264enc") !=
               cap.hw_encoders.end();
    if (!qsv) SKIP("no qsvh264enc encoder in this environment");
    const int W = 192, H = 144;
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::GStreamer;
    ecfg.hw_accel = HwAccel::Qsv;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = create_encoder_backend(ecfg);
    if (!enc) SKIP("no GStreamer encoder");
    auto gst = std::dynamic_pointer_cast<GstEncoder>(enc);
    std::string err;
    REQUIRE(enc->open("test_data/video/qsv_gst_out.mp4", W, H, 25, ecfg, &err));
    REQUIRE(gst != nullptr);
    REQUIRE(gst->used_hw());  // auto+Qsv 且 qsvh264enc 存在 → 必须真走 QSV 硬编，而非回退 x264
    uint8_t bgr[W * H * 3];
    memset(bgr, 128, sizeof(bgr));
    auto img = modeldeploy::vision::ImageData::from_raw(bgr, W, H, MdImageType::PKG_BGR_U8, true);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec != nullptr);
    REQUIRE(dec->open("test_data/video/qsv_gst_out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}
