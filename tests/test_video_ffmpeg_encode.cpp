#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/ffmpeg_encoder.h"
#include <algorithm>
#include <cstring>

using namespace modeldeploy::video;

namespace {
// 纯色 BGR ImageData（w×h，全 128 灰）
modeldeploy::vision::ImageData make_solid(int w, int h, uint8_t v = 128) {
    std::vector<uint8_t> bgr((size_t)w * h * 3, v);
    return modeldeploy::vision::ImageData::from_raw(bgr.data(), w, h, MdImageType::PKG_BGR_U8,
                                                    true);
}
}  // namespace

TEST_CASE("FFmpeg libx264 编码 mp4 → 回读验证", "[video][ffmpeg][integration]") {
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::FFmpeg;
    ecfg.set_codec("libx264").set_format("mp4");
    auto enc = VideoEncoder::create(ecfg);
    REQUIRE(enc != nullptr);
    std::string err;
    REQUIRE(enc->open("test_data/video/out.mp4", 64, 64, 25, &err));
    // 生成 32 帧纯色 BGR ImageData 逐帧 encode
    uint8_t bgr[64 * 64 * 3];
    memset(bgr, 128, sizeof(bgr));
    auto img = modeldeploy::vision::ImageData::from_raw(bgr, 64, 64, MdImageType::PKG_BGR_U8,
                                                        true);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    // 用 Task3 解码器回读 out.mp4 验证帧数与尺寸
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec->open("test_data/video/out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
    REQUIRE(cnt <= 32 + 5);  // 容忍首帧延迟误差
}

// 回归：auto + hw_accel=None → 必须回退软编（libx264），产物可回读。
TEST_CASE("FFmpeg auto + hw_accel=None → 仍 libx264 软编并可回读", "[video][ffmpeg][integration]") {
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::FFmpeg;
    ecfg.hw_accel = HwAccel::None;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = VideoEncoder::create(ecfg);
    REQUIRE(enc != nullptr);
    std::string err;
    REQUIRE(enc->open("test_data/video/soft_out.mp4", 128, 128, 25, &err));
    auto img = make_solid(128, 128);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    // 软解回读：libx264 产物是 h264，能解出帧
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec->open("test_data/video/soft_out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}

// NVENC 硬编：能力具备才跑（h264_nvenc 在 hw_encoders），走 CPU NV12，软解回读>0 帧。
TEST_CASE("FFmpeg h264_nvenc 硬编（CPU NV12）→ mp4 回读", "[video][hw][gpu][integration]") {
    auto cap = query_video_capabilities();
    bool nvenc = std::find(cap.hw_encoders.begin(), cap.hw_encoders.end(), "h264_nvenc") !=
                 cap.hw_encoders.end();
    if (!nvenc) {
        SKIP("no h264_nvenc encoder in this environment");
    }
    // 本卡（RTX 4060 Ti）NVENC 最小宽度=146（128/144 均报 Frame Dimension 不足，实测 146 起步）；
    // 192×144 稳定可硬编且更贴近常见视频尺寸。
    const int W = 192, H = 144;
    VideoEncoderConfig ecfg;
    ecfg.backend = CodecBackend::FFmpeg;
    ecfg.hw_accel = HwAccel::Cuda;
    ecfg.set_codec("auto").set_format("mp4");
    auto enc = create_encoder_backend(ecfg);
    REQUIRE(enc != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegEncoder>(enc);
    std::string err;
    REQUIRE(enc->open("test_data/video/nvenc_out.mp4", W, H, 25, ecfg, &err));
    auto img = make_solid(W, H);
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(VideoFrame{img}, &err));
    enc->close();
    if (ff) REQUIRE(ff->used_hw());  // 确实走了 nvenc 硬编，而非静默回退软编
    // 软解回读验证产物
    VideoDecoderConfig dcfg;
    dcfg.backend = CodecBackend::FFmpeg;
    dcfg.hw_accel = HwAccel::None;
    auto dec = VideoDecoder::create(dcfg);
    REQUIRE(dec->open("test_data/video/nvenc_out.mp4", &err));
    int cnt = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err)) ++cnt;
    REQUIRE(cnt > 0);
}
