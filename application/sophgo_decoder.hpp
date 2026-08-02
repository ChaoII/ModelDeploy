//
// Created by aichao on 2025/8/2.
// Sophgo 算能硬解码器：sophon-mw bm_video_decode（VPU/JM）解码 RTSP → bm_image（设备 NV12），
// 零拷贝直进 TPU 推理（经 SophgoProcessorBackend::process_device_image）。
// 仅在 ENABLE_SOPHGO 编译（Linux + sophon-mw SDK）。API 随版本有差异，标注 VERIFY。
//
#pragma once

#include <string>
#include <memory>
#include "stream_decoder.hpp"

namespace modeldeploy::app {

class SophgoDecoder {
public:
    explicit SophgoDecoder(const DecoderConfig& cfg);
    ~SophgoDecoder();

    bool open(const std::string& url);

    // 阻塞读一帧；返回的 DecodedFrame.device_image 为 bm_image*（设备 NV12）
    bool read_one_frame(DecodedFrame* out);

    void close();

private:
    DecoderConfig cfg_;
    std::string url_;
    // 不透明句柄：实际为 bm_video_decode*（见 .cpp，避免头文件引入 sophon-mw）
    void* decoder_ = nullptr;
    bool open_ = false;
};

} // namespace modeldeploy::app
