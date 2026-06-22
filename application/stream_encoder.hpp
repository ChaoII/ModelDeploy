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
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include "config.hpp"
#include "csrc/vision/common/image_data.h"

/// H.264 编码器（NVENC / x264） + RTSP/RTMP/MP4/FLV 输出
class StreamEncoder {
public:
    explicit StreamEncoder(const EncoderConfig& cfg = EncoderConfig{});
    ~StreamEncoder();

    /// 打开输出流（RTMP / RTSP / 本地文件）
    /// @param source_fps 源流帧率（cfg.fps=0 时使用此值自动匹配）
    bool open(const std::string& output_url, int width, int height, int source_fps = 25);

    /// 关闭输出流
    void close();

    /// 编码并推送一帧（ImageData 格式，BGR CPU）
    bool encode(const modeldeploy::vision::ImageData& image);

    /// 同步编码，返回编码耗时(us)
    int64_t encode_timed(const modeldeploy::vision::ImageData& image);

    /// 异步编码（放入队列，编码线程处理）
    bool encode_async(const modeldeploy::vision::ImageData& image);

    /// 启动异步编码线程
    bool start_async();

    /// 停止异步编码线程（会先排空队列）
    void stop_async();

    /// 是否已打开
    bool is_open() const { return opened_.load(); }

    /// 是否永久性推流失败（如地址已被占用）
    bool has_permanently_failed() const { return open_permanently_failed_.load(); }

    /// 获取最后的推流错误信息
    std::string last_error() const { return last_error_; }

    /// 本路流的目标帧率（用于 wall-clock PTS）
    int target_fps() const {
        int fps = cfg_.fps;
        return fps > 0 ? fps : 25;
    }

private:
    EncoderConfig cfg_;
    int width_ = 0, height_ = 0;
    std::atomic<bool> opened_{false};

    // FFmpeg encoder
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* enc_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVFrame* enc_frame_ = nullptr;     // 复用 AVFrame
    AVPacket* enc_pkt_ = nullptr;      // 复用 AVPacket
    int64_t frame_pts_ = 0;

    // Wall-clock PTS: 记录编码器启动时刻，PTS 对齐真实时间
    std::chrono::steady_clock::time_point encode_start_time_;
    int64_t encode_frame_count_ = 0;
    bool header_written_ = false;       // avformat_write_header 成功

    int sws_src_stride_ = 0;           // 缓存上次 sws 的 src stride，stride 变化时重建

    // 推流重试跟踪：地址被占用时停止重试
    std::atomic<bool> open_permanently_failed_{false};
    int open_retries_ = 0;
    std::string last_error_;

    // Async encoding
    std::atomic<bool> async_running_{false};
    std::atomic<bool> async_started_{false};
    std::thread encode_thread_;
    std::queue<modeldeploy::vision::ImageData> frame_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_ = 30;

    bool init_encoder(int width, int height);
    bool open_output(const std::string& url);
    void encode_loop();
    void drain_queue();
};
