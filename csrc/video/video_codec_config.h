#pragma once
#include "core/md_decl.h"
#include "video_common.h"
#include <string>

namespace modeldeploy::video {

// 编解码会话通用配置（解码/编码共用的网络与后端参数）
struct VideoCodecConfig {
    CodecBackend backend = CodecBackend::FFmpeg;
    HwAccel hw_accel = HwAccel::Auto;
    int reconnect_delay_ms = 5000;
    int max_reconnects = 10;
    int timeout_us = 10000000;
    std::string rtsp_transport = "tcp";
    bool device_only = false;
};

struct VideoDecoderConfig : VideoCodecConfig {
    MODELDEPLOY_CXX_EXPORT bool validate(std::string* err) const;
};

struct VideoEncoderConfig : VideoCodecConfig {
    int fps = 0;                 // 0=自动
    int bitrate_kbps = 2500;
    int gop = 12;
    std::string codec = "auto";  // auto/libx264/x264enc/h264_nvenc/nvh264enc/vaapih264enc
    std::string preset = "ultrafast";
    std::string format = "auto"; // auto/rtsp/rtmp/flv/mp4
    int max_b_frames = 0;
    bool low_latency = true;
    size_t async_queue_size = 30;

    VideoEncoderConfig& set_fps(int v) { fps = v; return *this; }
    VideoEncoderConfig& set_bitrate_kbps(int v) { bitrate_kbps = v; return *this; }
    VideoEncoderConfig& set_codec(const std::string& v) { codec = v; return *this; }
    VideoEncoderConfig& set_format(const std::string& v) { format = v; return *this; }
    MODELDEPLOY_CXX_EXPORT bool validate(std::string* err) const;
};

} // namespace modeldeploy::video
