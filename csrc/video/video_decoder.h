#pragma once

#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include <cstdint>
#include <string>

// FFmpeg C 头
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace modeldeploy::video {

// SDK 视频软解：复用 application/stream_decoder 的 FFmpeg 软解核心（h264/hevc 软解 → NV12）。
// YAGNI：只做 FFmpeg 软解，不做 CUVID/Sophgo/hw_accel（避免 SDK 引入 CUDA 依赖）。
class MODELDEPLOY_CXX_EXPORT VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder();

    bool open(const std::string& url);
    // 抽下一帧；成功时 *out 为 CPU NV12 ImageData（owner 保活解码帧），*pts_ms 为毫秒时间戳
    bool next(modeldeploy::vision::ImageData* out, uint64_t* pts_ms);
    void close();

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }

private:
    void cleanup();
    // 把 frame_（非 NV12）经 swscale 转成 NV12 存到 sws_frame_；成功返回 true
    bool convert_to_nv12();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* dec_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;    // 解码器输出帧（被 next 内 ref 到 owner）
    AVFrame* sws_frame_ = nullptr;  // swscale 转换目标（NV12），持有自有 buffer
    SwsContext* sws_ctx_ = nullptr; // swscale 上下文（源格式变化时重建）
    int sws_src_fmt_ = -1;          // sws_ctx_ 对应的源像素格式
    int width_ = 0;
    int height_ = 0;
    double fps_ = 25.0;
};

} // namespace modeldeploy::video
