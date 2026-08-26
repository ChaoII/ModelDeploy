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
    REQUIRE(hw_from_string("none") == mv::HwAccel::None);
    REQUIRE(hw_from_string("") == mv::HwAccel::None);
    REQUIRE(hw_from_string("auto") == mv::HwAccel::Auto);
    REQUIRE(hw_from_string("junk") == mv::HwAccel::Auto);
}

TEST_CASE("decoder config mapping", "[video_codec]") {
    DecoderConfig dc;
    dc.reconnect_delay_ms = 1000;
    dc.max_reconnects = 3;
    dc.timeout_us = 5000000;
    dc.rtsp_transport = "udp";
    dc.hw_accel = "cuda";
    dc.device_only = true;
    modeldeploy::video::VideoDecoderConfig sdk;
    video_codec_fill_decoder(&sdk, dc);
    REQUIRE(sdk.reconnect_delay_ms == 1000);
    REQUIRE(sdk.max_reconnects == 3);
    REQUIRE(sdk.timeout_us == 5000000);
    REQUIRE(sdk.rtsp_transport == "udp");
    REQUIRE(sdk.hw_accel == modeldeploy::video::HwAccel::Cuda);
    REQUIRE(sdk.device_only == true);
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

    modeldeploy::video::VideoEncoderConfig cpu;
    video_codec_fill_encoder(&cpu, ec, false);
    REQUIRE(cpu.hw_accel == modeldeploy::video::HwAccel::None);
    REQUIRE(cpu.gpu_direct_input == false);
}
