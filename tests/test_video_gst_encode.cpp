#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_encoder.h"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
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
    for (int i = 0; i < 32; ++i) REQUIRE(enc->encode(img, &err));
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
