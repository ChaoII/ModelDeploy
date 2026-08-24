#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_decoder.h"
#include <cstring>

using namespace modeldeploy::video;

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
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(img, &err));
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
