//
// Created by aichao on 2025/8/2.
// Sophgo 硬解码实现。
//
// 说明：sophon-mw 的 bm_video_decode API 随 SDK 版本有差异，下列调用基于 BM1688/CV186AH
// sophon-mw v2.x 文档编写，真机联调时需按实际头文件核对（VERIFY 点）。
// 未安装 SDK（ENABLE_SOPHGO off）时 open() 返回 false，应用回退 FFmpeg 软解。
//

#include "sophgo_decoder.hpp"
#include <iostream>

#ifdef ENABLE_SOPHGO
#include "bm_video_decode.h"

namespace modeldeploy::app {
    SophgoDecoder::SophgoDecoder(const DecoderConfig& cfg) : cfg_(cfg) {}

    SophgoDecoder::~SophgoDecoder() {
        close();
    }

    bool SophgoDecoder::open(const std::string& url) {
        if (open_) close();
        url_ = url;
        // VERIFY: bm_video_decode 构造函数签名（版本差异）
        // 常见形式：bm_video_decode(url, stream_buffer_size, read_frame_interval)
        try {
            constexpr int kStreamBufferSize = 64 * 1024 * 1024;
            auto* dec = new bm_video_decode(url_.c_str(), kStreamBufferSize, 0);
            decoder_ = dec;
            open_ = true;
            std::cout << "[SophgoDecoder] Opened (hardware VPU): " << url_ << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[SophgoDecoder] Open failed: " << e.what() << std::endl;
            close();
            return false;
        }
    }

    bool SophgoDecoder::read_one_frame(DecodedFrame* out) {
        if (!decoder_) return false;
        // VERIFY: bm_video_decode::get_frame 签名
        auto* dec = static_cast<bm_video_decode*>(decoder_);
        bm_image* frame = nullptr;  // VERIFY: get_frame(bm_image&, timeout) 或返回指针
        const int ret = dec->get_frame(frame, cfg_.timeout_us);
        if (ret != 0) return false;

        out->device_image = frame;  // bm_image*，供 process_device_image 零拷贝
        out->width = frame->width;
        out->height = frame->height;
        out->y_step = 0;   // 设备侧无 CPU stride
        out->uv_step = 0;
        out->y_plane = nullptr;
        out->uv_plane = nullptr;
        out->pts = 0;
        return true;
    }

    void SophgoDecoder::close() {
        if (decoder_) {
            delete static_cast<bm_video_decode*>(decoder_);
            decoder_ = nullptr;
        }
        open_ = false;
    }
} // namespace modeldeploy::app

#else // !ENABLE_SOPHGO

namespace modeldeploy::app {
    SophgoDecoder::SophgoDecoder(const DecoderConfig& cfg) : cfg_(cfg) {
        std::cerr << "[SophgoDecoder] ENABLE_SOPHGO off, hardware decode unavailable."
            << std::endl;
    }
    SophgoDecoder::~SophgoDecoder() { close(); }
    bool SophgoDecoder::open(const std::string&) { return false; }
    bool SophgoDecoder::read_one_frame(DecodedFrame*) { return false; }
    void SophgoDecoder::close() {}
} // namespace modeldeploy::app

#endif // ENABLE_SOPHGO
