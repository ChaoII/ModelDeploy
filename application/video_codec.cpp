#include "video_codec.hpp"
using modeldeploy::video::HwAccel;
using modeldeploy::video::CodecBackend;
using modeldeploy::video::Backpressure;
using modeldeploy::video::VideoDecoderConfig;
using modeldeploy::video::VideoEncoderConfig;

EncodeTopology parse_topology(const std::string& s) {
    if (s == "mosaic") return EncodeTopology::Mosaic;
    if (s == "both")   return EncodeTopology::Both;
    return EncodeTopology::PerChannel;
}

HwAccel hw_from_string(const std::string& dev) {
    if (dev == "cuda") return HwAccel::Cuda;
    if (dev == "vaapi") return HwAccel::Vaapi;
    if (dev == "qsv") return HwAccel::Qsv;
    if (dev == "sophgo") return HwAccel::Sophgo;
    if (dev == "none" || dev.empty()) return HwAccel::None;
    return HwAccel::Auto;
}

CodecBackend backend_from_string(const std::string& b) {
    if (b == "gstreamer") return CodecBackend::GStreamer;
    if (b == "ffmpeg") return CodecBackend::FFmpeg;
    return CodecBackend::Auto;
}

Backpressure backpressure_from_string(const std::string& s) {
    if (s == "drop") return Backpressure::Drop;
    if (s == "overwrite" || s == "overwrite_oldest") return Backpressure::OverwriteOldest;
    return Backpressure::Block;
}

void video_codec_fill_decoder(VideoDecoderConfig* sdk, const DecoderConfig& dc) {
    sdk->reconnect_delay_ms = dc.reconnect_delay_ms;
    sdk->max_reconnects = dc.max_reconnects;
    sdk->timeout_us = dc.timeout_us;
    sdk->rtsp_transport = dc.rtsp_transport;
    sdk->hw_accel = hw_from_string(dc.hw_accel);
    sdk->backend = backend_from_string(dc.backend);
    sdk->device_only = dc.device_only;
    sdk->codec = dc.codec;
    sdk->async_queue_size = dc.async_queue_size;
    sdk->pooling = dc.pooling;
    sdk->backpressure = backpressure_from_string(dc.backpressure);
}

void video_codec_fill_encoder(VideoEncoderConfig* sdk, const EncoderConfig& ec, bool gpu_direct) {
    sdk->fps = ec.fps;
    sdk->bitrate_kbps = ec.bitrate_kbps;
    sdk->gop = ec.gop;
    sdk->codec = ec.codec;
    sdk->backend = backend_from_string(ec.backend);
    sdk->preset = ec.preset;
    sdk->format = ec.format;
    sdk->max_b_frames = ec.max_b_frames;
    sdk->low_latency = ec.low_latency;
    sdk->device_only = ec.device_only;
    sdk->async_queue_size = ec.async_queue_size;
    sdk->pooling = ec.pooling;
    sdk->backpressure = backpressure_from_string(ec.backpressure);
    // GPU 直编：强制 cuda；否则按配置（auto 时 SDK 自行选择 nvenc/qsv/vaapi/x264）
    sdk->hw_accel = gpu_direct ? HwAccel::Cuda : hw_from_string(ec.hw_accel);
    sdk->gpu_direct_input = gpu_direct || ec.gpu_direct_input;
}
