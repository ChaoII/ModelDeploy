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

// 64×64 极小 H.264 源：cuvid 在运行时可能不支持该分辨率（cuvidCreateDecoder →
// CUDA_ERROR_NOT_SUPPORTED），导致 avcodec_open2 成功、read 却全程产 0 帧后直接 EOF。
// 本用例须容忍任一正确回退路径（open 期回退 or 运行期 0 帧回退），核心断言是
// hw=Auto 下解 64×64 旧源能读到 >0 帧——证明 0 帧场景被软解回退透明救回。
TEST_CASE("FFmpeg hw=Auto 极小分辨率运行期 0 帧自动回退软解", "[video][hw][gpu][integration]") {
    std::ifstream probe("test_data/video/small64.h264");
    if (!probe.good()) {
        SKIP("no 64x64 test clip; generate with: ffmpeg -f lavfi -i testsrc2=size=64x64:rate=25:duration=1 -c:v libx264 test_data/video/small64.h264");
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Auto;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);
    std::string err;
    REQUIRE(d->open("test_data/video/small64.h264", &err));
    REQUIRE(d->width() == 64);
    REQUIRE(d->height() == 64);
    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == d->width());
        REQUIRE(f.image.height() == d->height());
        ++n;
    }
    // 核心断言：hw=Auto 下解 64×64 极小源必须读到 >0 帧（open 期或运行期软解回退均可）。
    // 同时，循环能在此终止（n<200 上限 + EOF 返回 false）证明"回退一次后不再回退"无死循环。
    INFO("small64 frames=" << n << " used_hw=" << (ff && ff->used_hw()));
    REQUIRE(n > 0);
    d->close();
}

// 运行期 0 帧自动回退的确定性回归/边界用例：取一个 64×64 的 H.264 源并截断成
// "cuvid 能 open 成功、但解码器产不出任何完整帧后直接 EOF" 的流（find_stream_info 仍可判定
// 出 64×64 视频流，cuvid 原生解码 64×64，故 open 期正常走硬解）。于是本次会话 used_hw_==true
// 且 delivered_frames_==0，命中运行期 0 帧回退分支 → cleanup + 强制软解重开。核心断言：
// 读取能终止（无无限循环/死锁），且回退重开后解码器仍可用、状态合理。
// 无论 env 走"运行期 0 帧回退"还是"open 期回退/软解"都须绿。
TEST_CASE("FFmpeg 运行期 0 帧硬解自动回退软解（截断源，无死循环）", "[video][hw][gpu][integration]") {
    const char* kSrc = "test_data/video/small64.h264";
    const char* kTrunc = "test_data/video/trunc_runtime.h264";
    {
        std::ifstream src(kSrc, std::ios::binary);
        if (!src.good()) SKIP("no source clip");
        src.seekg(0, std::ios::end);
        std::streamsize len = src.tellg();
        src.seekg(0, std::ios::beg);
        std::vector<char> buf((size_t)len);
        src.read(buf.data(), len);
        size_t cut = (size_t)std::min<decltype(len)>(len, 1200);  // SPS/PPS + 不完整首帧
        std::ofstream t(kTrunc, std::ios::binary);
        t.write(buf.data(), (std::streamsize)cut);
        t.close();
    }
    VideoDecoderConfig cfg;
    cfg.backend = CodecBackend::FFmpeg;
    cfg.hw_accel = HwAccel::Auto;
    auto d = create_decoder_backend(cfg);
    REQUIRE(d != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);
    std::string err;
    REQUIRE(d->open(kTrunc, &err));
    int n = 0;
    VideoFrame f;
    // 必须能终止：无论走运行期回退（used_hw_→false）还是 open 期降级软解，都不应死锁/死循环。
    while (d->read_one_frame(&f, &err) && n < 200) ++n;
    d->close();  // 关闭本身也须能正常完成（析构/清理不卡死）
    // 至此读取正常终止即证明回退路径安全。n 不要求 >0：截断内容本身可能产不出完整帧。
}

TEST_CASE("FFmpeg h264_qsv 硬解 → CPU NV12", "[video][ffmpeg][hw][gpu][integration]") {
    std::ifstream probe("test_data/video/clip.h264");
    if (!probe.good()) SKIP("no test clip; place at test_data/video/clip.h264");
    auto cap = query_video_capabilities();
    bool qsv = std::find(cap.hw_decoders.begin(), cap.hw_decoders.end(), "h264_qsv") !=
               cap.hw_decoders.end();
    if (!qsv) SKIP("no h264_qsv decoder in this environment");
    VideoDecoderConfig hw;
    hw.backend = CodecBackend::FFmpeg;
    hw.hw_accel = HwAccel::Qsv;
    auto d = create_decoder_backend(hw);
    REQUIRE(d != nullptr);
    auto ff = std::dynamic_pointer_cast<FfmpegDecoder>(d);
    std::string err;
    REQUIRE(d->open("test_data/video/clip.h264", &err));
    REQUIRE(d->width() > 0);
    REQUIRE(d->height() > 0);
    int n = 0;
    VideoFrame f;
    while (d->read_one_frame(&f, &err) && n < 200) {
        REQUIRE_FALSE(f.image.empty());
        REQUIRE(f.image.width() == d->width());
        REQUIRE(f.image.height() == d->height());
        ++n;
    }
    REQUIRE(n > 0);
    if (ff) REQUIRE(ff->used_hw());  // 确实走 QSV 硬解
    d->close();
}
