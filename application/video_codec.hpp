#pragma once
#include <string>
#include "config.hpp"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"

enum class EncodeTopology { PerChannel, Mosaic, Both };

EncodeTopology parse_topology(const std::string& s);
modeldeploy::video::HwAccel hw_from_string(const std::string& dev);
void video_codec_fill_decoder(modeldeploy::video::VideoDecoderConfig* sdk, const DecoderConfig& dc);
void video_codec_fill_encoder(modeldeploy::video::VideoEncoderConfig* sdk, const EncoderConfig& ec, bool gpu_direct);
