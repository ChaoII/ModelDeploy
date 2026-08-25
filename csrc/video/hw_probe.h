#pragma once
#include <string>
#include <vector>

namespace modeldeploy::video {

// 硬件编解码能力自动探测结果：仅向使用者暴露名称字符串，不泄漏 FFmpeg/GStreamer 原生结构。
struct HwProbeResult {
    std::vector<std::string> decoders;  // 如 "h264_cuvid"
    std::vector<std::string> encoders;  // 如 "h264_nvenc"
};

// 探测编译/运行环境下可用的 FFmpeg 硬件编解码器（CUVID/NVENC 需 CUDA 设备上下文可建才计入）。
// ENABLE_FFMPEG 关闭时返回空结果。
HwProbeResult probe_ffmpeg_hw();
// 探测编译/运行环境下可用的 GStreamer 硬件编解码器插件。ENABLE_GSTREAMER 关闭时返回空结果。
HwProbeResult probe_gstreamer_hw();

} // namespace modeldeploy::video
