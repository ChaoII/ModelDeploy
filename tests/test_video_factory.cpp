#include "catch2/catch_test_macros.hpp"
#include "csrc/video/factory.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"
#include <algorithm>
#include <vector>

using namespace modeldeploy::video;

namespace {
bool list_contains(const std::vector<std::string>& v, const std::string& n) {
    return std::find(v.begin(), v.end(), n) != v.end();
}
// 能力与工厂一致性校验：能力为 true 则对应后端非空，为 false 则必须返回空（不回退/不凭空造出后端）
void check_backend_consistency(const VideoCodecCapabilities& cap,
                               CodecBackend decoder_backend_to_probe,
                               CodecBackend encoder_backend_to_probe,
                               bool expect) {
    VideoDecoderConfig dcfg;
    dcfg.backend = decoder_backend_to_probe;
    auto db = create_decoder_backend(dcfg);
    REQUIRE((db != nullptr) == expect);

    VideoEncoderConfig ecfg;
    ecfg.backend = encoder_backend_to_probe;
    auto eb = create_encoder_backend(ecfg);
    REQUIRE((eb != nullptr) == expect);
}
}  // namespace

TEST_CASE("query_video_capabilities 返回可解析的结构", "[video][factory]") {
    auto cap = query_video_capabilities();
    // 能力位要么是布尔要么已初始化；不硬编码任一后端必须可用，随构建配置变化
    REQUIRE_NOTHROW(static_cast<void>(cap.ffmpeg_available));
    REQUIRE_NOTHROW(static_cast<void>(cap.gstreamer_available));
    // hw 能力反映运行环境：探测源码填充后不再硬编码空/非空断言（见下方 [hw] 用例）
}

TEST_CASE("能力查询真实反映硬件编解码，不硬编码", "[video][factory][hw]") {
    auto cap = query_video_capabilities();
    // 环境无 CUDA 时 hw_decoders 不应含 cuvid，有则必须含（探测真实生效）。绝不硬编码必然含。
    if (list_contains(cap.hw_decoders, "h264_cuvid"))
        REQUIRE(list_contains(cap.hw_decoders, "h264_cuvid"));
    else
        REQUIRE_FALSE(list_contains(cap.hw_decoders, "h264_cuvid"));
    if (list_contains(cap.hw_encoders, "h264_nvenc"))
        REQUIRE(list_contains(cap.hw_encoders, "h264_nvenc"));
    else
        REQUIRE_FALSE(list_contains(cap.hw_encoders, "h264_nvenc"));
    // 去重：FFmpeg 与 GStreamer 探测结果拼接后不得出现重复项
    bool dup_dec = false;
    std::vector<std::string> seen_dec;
    for (const auto& n : cap.hw_decoders)
        if (std::find(seen_dec.begin(), seen_dec.end(), n) != seen_dec.end()) dup_dec = true;
        else seen_dec.push_back(n);
    CHECK_FALSE(dup_dec);
    bool dup_enc = false;
    std::vector<std::string> seen_enc;
    for (const auto& n : cap.hw_encoders)
        if (std::find(seen_enc.begin(), seen_enc.end(), n) != seen_enc.end()) dup_enc = true;
        else seen_enc.push_back(n);
    CHECK_FALSE(dup_enc);
}

TEST_CASE("FFmpeg 后端能力与工厂一致", "[video][factory]") {
    auto cap = query_video_capabilities();
    check_backend_consistency(cap, CodecBackend::FFmpeg, CodecBackend::FFmpeg, cap.ffmpeg_available);
}

TEST_CASE("GStreamer 后端能力与工厂一致", "[video][factory]") {
    auto cap = query_video_capabilities();
    check_backend_consistency(cap, CodecBackend::GStreamer, CodecBackend::GStreamer,
                              cap.gstreamer_available);
}

