#pragma once
#include "core/md_decl.h"
#include "video_common.h"
#include <string>

namespace modeldeploy::video {

// 有界异步队列满时的背压策略（双后端统一在门面/路由层实现）
enum class Backpressure { Block, Drop, OverwriteOldest };

// 编解码会话通用配置（解码/编码共用的网络与后端参数）
struct VideoCodecConfig {
    CodecBackend backend = CodecBackend::FFmpeg;
    HwAccel hw_accel = HwAccel::Auto;
    int reconnect_delay_ms = 5000;
    int max_reconnects = 10;
    int timeout_us = 10000000;
    std::string rtsp_transport = "tcp";
    bool device_only = false;
    // 异步解码队列容量（有界）；满时按 backpressure 处理
    int async_queue_size = 30;
    // 帧缓冲池总开关：true 复用 VideoFrame 容器，false 每帧新建
    bool pooling = true;
    // 有界队列满时的背压策略
    Backpressure backpressure = Backpressure::Block;
};

struct VideoDecoderConfig : VideoCodecConfig {
    // 请求编解码格式（默认自动）：auto/hevc_qsv/qsvh265dec/h264_qsv/qsvh264dec 等。
    // FFmpeg 解码自动探测码流；GStreamer QSV 依赖它区分 HEVC(qsvh265dec+h265parse) 与 H.264(qsvh264dec+h264parse)。
    std::string codec = "auto";
    VideoDecoderConfig& set_codec(const std::string& v) { codec = v; return *this; }
    MODELDEPLOY_CXX_EXPORT bool validate(std::string* err) const;
};

struct VideoEncoderConfig : VideoCodecConfig {
    int fps = 0;                 // 0=自动
    int bitrate_kbps = 2500;
    int gop = 12;
    std::string codec = "auto";  // auto/libx264/x264enc/h264_nvenc/nvh264enc/vaapih264enc/h264_bm/h265_bm/h264_qsv/hevc_qsv/qsvh264enc/qsvh265enc
    std::string preset = "ultrafast";
    std::string format = "auto"; // auto/rtsp/rtmp/flv/mp4
    int max_b_frames = 0;
    bool low_latency = true;
    // GPU 直接编码：true 时（配合 hw_accel=Cuda 且 nvenc/nvh264enc）encode(const VideoFrame&)
    // 以 device=GPU 的设备 NV12 直编，不做主机往返。仅设备路径使用，CPU encode() 不受影响。
    bool gpu_direct_input = false;

    VideoEncoderConfig& set_fps(int v) { fps = v; return *this; }
    VideoEncoderConfig& set_bitrate_kbps(int v) { bitrate_kbps = v; return *this; }
    VideoEncoderConfig& set_codec(const std::string& v) { codec = v; return *this; }
    VideoEncoderConfig& set_format(const std::string& v) { format = v; return *this; }
    MODELDEPLOY_CXX_EXPORT bool validate(std::string* err) const;
};

} // namespace modeldeploy::video
