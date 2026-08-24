#pragma once
#include "csrc/video/video_codec_config.h"
#include "core/md_decl.h"
#include <memory>
#include <string>
#include <vector>

namespace modeldeploy::video {
class DecoderBackend;
class EncoderBackend;

// 视频编解码能力探测结果
struct VideoCodecCapabilities {
    bool ffmpeg_available = false;
    bool gstreamer_available = false;
    std::vector<std::string> hw_decoders;   // 如 "h264_cuvid"
    std::vector<std::string> hw_encoders;   // 如 "h264_nvenc"
};

// 探测当前编译/运行环境下可用的视频后端与硬件编解码能力
MODELDEPLOY_CXX_EXPORT VideoCodecCapabilities query_video_capabilities();
// 依据配置创建解码后端；无法创建（后端未启用/不可用）时返回 nullptr
MODELDEPLOY_CXX_EXPORT std::shared_ptr<DecoderBackend> create_decoder_backend(const VideoDecoderConfig& cfg);
// 依据配置创建编码后端；无法创建（后端未启用/不可用）时返回 nullptr
MODELDEPLOY_CXX_EXPORT std::shared_ptr<EncoderBackend> create_encoder_backend(const VideoEncoderConfig& cfg);
} // namespace modeldeploy::video
