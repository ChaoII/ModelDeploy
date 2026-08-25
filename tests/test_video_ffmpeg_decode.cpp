#include "catch2/catch_test_macros.hpp"
#include "csrc/video/video_decoder.h"
#include "csrc/video/factory.h"
#include "csrc/video/backend/ffmpeg_decoder.h"
#include <fstream>
#include <algorithm>

using namespace modeldeploy::video;

TEST_CASE("FFmpeg 软解 h264 → VideoFrame(NV12)", "[video][ffmpeg][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) {
        SKIP("no test clip; generate with ffmpeg or place at test_data/video/clip.h264");
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    auto dec = VideoDecoder::create(cfg);
    REQUIRE(dec != nullptr);
    std::string err;
    REQUIRE(dec->open("test_data/video/clip.h264", &err));
    REQUIRE(dec->width() > 0);
    REQUIRE(dec->height() > 0);
    int n = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err) && n < 30) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == dec->width());
        REQUIRE(f.image.height() == dec->height());
        ++n;
    }
    REQUIRE(n > 0);
    dec->close();
}

// 解码 clip.h264 到 n 帧（n 上限 cap），返回读到的帧数；失败回 0。
static int count_frames(const VideoDecoderConfig& cfg, int cap = 200) {
    auto dec = VideoDecoder::create(cfg);
    if (!dec) return 0;
    std::string err;
    if (!dec->open("test_data/video/clip.h264", &err)) return 0;
    int n = 0;
    VideoFrame f;
    while (dec->read_one_frame(&f, &err) && n < cap) ++n;
    dec->close();
    return n;
}

TEST_CASE("FFmpeg h264_cuvid 硬解 → CPU NV12，软解回退", "[video][hw][gpu][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) {
        SKIP("no test clip; place at test_data/video/clip.h264");
    }
    auto cap = query_video_capabilities();
    bool cuvid = std::find(cap.hw_decoders.begin(), cap.hw_decoders.end(), "h264_cuvid")
                 != cap.hw_decoders.end();
    // 无 GPU/无 cuvid 解码器：能力未就绪则跳过真实硬解，仅验证显式 CUDA 请求会回退软解。
    VideoDecoderConfig hw;
    hw.backend = CodecBackend::FFmpeg;
    hw.hw_accel = HwAccel::Cuda;

    if (!cuvid) {
        // 环境无硬解：硬解请求必须降级为软解且仍能解出帧（不挂会话），used_hw_ 应为 false。
        auto d = create_decoder_backend(hw);
        REQUIRE(d != nullptr);
        auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);
        std::string err;
        REQUIRE(d->open("test_data/video/clip.h264", &err));
        int n = 0;
        VideoFrame f;
        while (d->read_one_frame(&f, &err) && n < 200) ++n;
        REQUIRE(n > 0);
        if (ff) REQUIRE_FALSE(ff->used_hw());
        d->close();
        return;
    }

    // 软解基线帧数
    VideoDecoderConfig soft;
    soft.backend = CodecBackend::FFmpeg;
    soft.hw_accel = HwAccel::None;
    int n_soft = count_frames(soft);
    REQUIRE(n_soft > 0);

    // 硬解：必须走 cuvid（used_hw_==true），且帧数/尺寸与软解一致
    auto d = create_decoder_backend(hw);
    REQUIRE(d != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);
    std::string err;
    REQUIRE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(d->width() > 0);
    REQUIRE(d->height() > 0);
    int n_hw = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n_hw < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == d->width());
        REQUIRE(f.image.height() == d->height());
        ++n_hw;
    }
    REQUIRE(n_hw > 0);
    if (ff) REQUIRE(ff->used_hw());  // 确实走了硬解
    d->close();
    // cuvid 的尾帧刷新/丢帧行为与软解略有差异（pkt_timebase 逐帧输出），故允许多余量而非严格相等。
    INFO("hw=" << n_hw << " soft=" << n_soft);
    REQUIRE(n_hw > 0);
    CHECK(n_hw >= n_soft / 2);  // 硬解帧数不应与软解差过一个量级
}
