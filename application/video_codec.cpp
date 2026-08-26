#include "video_codec.hpp"
using modeldeploy::video::HwAccel;
using modeldeploy::video::VideoDecoderConfig;
using modeldeploy::video::VideoEncoderConfig;

EncodeTopology parse_topology(const std::string& s) {
    if (s == "mosaic") return EncodeTopology::Mosaic;
    if (s == "both")   return EncodeTopology::Both;
    return EncodeTopology::PerChannel;
}

HwAccel hw_from_string(const std::string& dev) {
    if (dev == "cuda") return HwAccel::Cuda;
    if (dev == "none" || dev.empty()) return HwAccel::None;
    return HwAccel::Auto;
}

void video_codec_fill_decoder(VideoDecoderConfig* sdk, const DecoderConfig& dc) {
    sdk->reconnect_delay_ms = dc.reconnect_delay_ms;
    sdk->max_reconnects = dc.max_reconnects;
    sdk->timeout_us = dc.timeout_us;
    sdk->rtsp_transport = dc.rtsp_transport;
    sdk->hw_accel = hw_from_string(dc.hw_accel);
    sdk->device_only = dc.device_only;
}

void video_codec_fill_encoder(VideoEncoderConfig* sdk, const EncoderConfig& ec, bool gpu_direct) {
    sdk->fps = ec.fps;
    sdk->bitrate_kbps = ec.bitrate_kbps;
    sdk->gop = ec.gop;
    sdk->codec = ec.codec;
    sdk->preset = ec.preset;
    sdk->format = ec.format;
    sdk->max_b_frames = ec.max_b_frames;
    sdk->low_latency = ec.low_latency;
    sdk->hw_accel = gpu_direct ? HwAccel::Cuda : HwAccel::None;
    sdk->gpu_direct_input = gpu_direct;
}
