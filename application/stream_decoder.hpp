#pragma once
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

#include "config.hpp"

/// 解码帧回调：GPU NV12 帧的 Y 和 UV 设备指针、步长、宽高
struct DecodedFrame {
    uint8_t* y_plane = nullptr;
    uint8_t* uv_plane = nullptr;
    int width = 0;
    int height = 0;
    int y_step = 0;
    int uv_step = 0;
    int64_t pts = 0;
};

/// FFmpeg CUVID 硬件解码器
class StreamDecoder {
public:
    explicit StreamDecoder(const DecoderConfig& cfg);
    ~StreamDecoder();

    /// 打开视频流（RTSP / 本地文件）
    bool open(const std::string& url);

    /// 注册帧回调（在解码线程中调用）
    using FrameCallback = std::function<bool(const DecodedFrame&)>;
    void set_callback(FrameCallback cb) { callback_ = std::move(cb); }

    /// 启动解码线程（异步）
    bool start();

    /// 停止解码线程
    void stop();

    /// 是否正在运行
    bool is_running() const { return running_.load(); }

    /// 获取当前帧率
    int fps() const { return fps_; }

private:
    DecoderConfig cfg_;
    std::string url_;
    FrameCallback callback_;

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* dec_ctx_ = nullptr;
    AVBufferRef* hw_dev_ctx_ = nullptr;
    int video_stream_idx_ = -1;

    std::atomic<bool> running_{false};
    std::thread decode_thread_;
    int fps_ = 25;

    bool init_decoder();
    void decode_loop();
    void reconnect();
    void cleanup();
};
