#include <catch2/catch_test_macros.hpp>
#include "video_codec.hpp"
#include "config.hpp"

TEST_CASE("EncodeTopology parse", "[video_codec]") {
    REQUIRE(parse_topology("per_channel") == EncodeTopology::PerChannel);
    REQUIRE(parse_topology("mosaic") == EncodeTopology::Mosaic);
    REQUIRE(parse_topology("both") == EncodeTopology::Both);
    REQUIRE(parse_topology("bogus") == EncodeTopology::PerChannel);
    REQUIRE(parse_topology("") == EncodeTopology::PerChannel);
}

TEST_CASE("HwAccel map", "[video_codec]") {
    namespace mv = modeldeploy::video;
    REQUIRE(hw_from_string("cuda") == mv::HwAccel::Cuda);
    REQUIRE(hw_from_string("vaapi") == mv::HwAccel::Vaapi);
    REQUIRE(hw_from_string("qsv") == mv::HwAccel::Qsv);
    REQUIRE(hw_from_string("sophgo") == mv::HwAccel::Sophgo);
    REQUIRE(hw_from_string("none") == mv::HwAccel::None);
    REQUIRE(hw_from_string("") == mv::HwAccel::None);
    REQUIRE(hw_from_string("auto") == mv::HwAccel::Auto);
    REQUIRE(hw_from_string("junk") == mv::HwAccel::Auto);
}

TEST_CASE("Backend map", "[video_codec]") {
    namespace mv = modeldeploy::video;
    REQUIRE(backend_from_string("gstreamer") == mv::CodecBackend::GStreamer);
    REQUIRE(backend_from_string("ffmpeg") == mv::CodecBackend::FFmpeg);
    REQUIRE(backend_from_string("") == mv::CodecBackend::Auto);
    REQUIRE(backend_from_string("auto") == mv::CodecBackend::Auto);
    REQUIRE(backend_from_string("junk") == mv::CodecBackend::Auto);
}

TEST_CASE("decoder config mapping", "[video_codec]") {
    DecoderConfig dc;
    dc.reconnect_delay_ms = 1000;
    dc.max_reconnects = 3;
    dc.timeout_us = 5000000;
    dc.rtsp_transport = "udp";
    dc.hw_accel = "qsv";
    dc.backend = "gstreamer";
    dc.device_only = true;
    dc.codec = "hevc_qsv";
    dc.async_queue_size = 42;
    dc.pooling = false;
    dc.backpressure = "drop";
    modeldeploy::video::VideoDecoderConfig sdk;
    video_codec_fill_decoder(&sdk, dc);
    REQUIRE(sdk.reconnect_delay_ms == 1000);
    REQUIRE(sdk.max_reconnects == 3);
    REQUIRE(sdk.timeout_us == 5000000);
    REQUIRE(sdk.rtsp_transport == "udp");
    REQUIRE(sdk.hw_accel == modeldeploy::video::HwAccel::Qsv);
    REQUIRE(sdk.backend == modeldeploy::video::CodecBackend::GStreamer);
    REQUIRE(sdk.device_only == true);
    REQUIRE(sdk.codec == "hevc_qsv");
    REQUIRE(sdk.async_queue_size == 42);
    REQUIRE(sdk.pooling == false);
    REQUIRE(sdk.backpressure == modeldeploy::video::Backpressure::Drop);
}

TEST_CASE("encoder config mapping gpu vs cpu", "[video_codec]") {
    EncoderConfig ec;
    ec.bitrate_kbps = 4000;
    ec.codec = "h264_nvenc";
    ec.format = "flv";
    modeldeploy::video::VideoEncoderConfig gpu;
    video_codec_fill_encoder(&gpu, ec, true);
    REQUIRE(gpu.bitrate_kbps == 4000);
    REQUIRE(gpu.codec == "h264_nvenc");
    REQUIRE(gpu.format == "flv");
    REQUIRE(gpu.hw_accel == modeldeploy::video::HwAccel::Cuda);
    REQUIRE(gpu.gpu_direct_input == true);

    EncoderConfig ec2 = ec;
    ec2.hw_accel = "none";
    modeldeploy::video::VideoEncoderConfig cpu;
    video_codec_fill_encoder(&cpu, ec2, false);
    REQUIRE(cpu.hw_accel == modeldeploy::video::HwAccel::None);
    REQUIRE(cpu.gpu_direct_input == false);

    EncoderConfig ec3 = ec;
    ec3.hw_accel = "auto";
    ec3.backpressure = "overwrite";
    modeldeploy::video::VideoEncoderConfig auto_cfg;
    video_codec_fill_encoder(&auto_cfg, ec3, false);
    REQUIRE(auto_cfg.hw_accel == modeldeploy::video::HwAccel::Auto);
    REQUIRE(auto_cfg.backpressure == modeldeploy::video::Backpressure::OverwriteOldest);
}
