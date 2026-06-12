#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include "config.hpp"
#include "csrc/vision/common/image_data.h"

/// NVENC 编码器 + RTSP 推流
class StreamEncoder {
public:
    explicit StreamEncoder(int fps = 25);
    ~StreamEncoder();

    /// 打开输出流（RTMP / RTSP / 本地文件）
    bool open(const std::string& output_url, int width, int height);

    /// 关闭输出流
    void close();

    /// 编码并推送一帧（ImageData 格式，支持 BGR/NV12）
    bool encode(const modeldeploy::vision::ImageData& image);

    /// 异步编码（放入队列，编码线程处理）
    bool encode_async(const modeldeploy::vision::ImageData& image);

    /// 启动异步编码线程
    bool start_async();

    /// 停止异步编码线程
    void stop_async();

    /// 是否已打开
    bool is_open() const { return opened_.load(); }

private:
    int fps_;
    int width_ = 0, height_ = 0;
    std::atomic<bool> opened_{false};

    // FFmpeg encoder
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* enc_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int64_t frame_pts_ = 0;

    // Async encoding
    std::atomic<bool> async_running_{false};
    std::thread encode_thread_;
    std::queue<modeldeploy::vision::ImageData> frame_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_ = 30;

    bool init_encoder(int width, int height);
    bool open_output(const std::string& url);
    bool encode_frame(AVFrame* frame);
    void encode_loop();
};
