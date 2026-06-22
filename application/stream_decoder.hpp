#pragma once
#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

#include "config.hpp"

/// 解码帧回调：NV12 帧的 Y 和 UV 指针、步长、宽高
struct DecodedFrame {
    uint8_t* y_plane = nullptr;
    uint8_t* uv_plane = nullptr;
    int width = 0;
    int height = 0;
    int y_step = 0;
    int uv_step = 0;
    int64_t pts = 0;
};

/// FFmpeg 硬件/软件解码器，支持 CUVID 回退与自动重连
class StreamDecoder {
public:
    explicit StreamDecoder(const DecoderConfig& cfg);
    ~StreamDecoder();

    /// 打开视频流（RTSP / 本地文件）
    bool open(const std::string& url);

    /// 同步解码一帧（阻塞直到返回一帧/EOF/错误）
    /// decoder 线程不再单独存在，由 pipeline 线程直接调用
    /// @return true 表示有帧，false 表示 EOF 或错误
    bool read_one_frame(DecodedFrame* out);

    /// 注册帧回调（异步模式，不启用时忽略）
    using FrameCallback = std::function<bool(const DecodedFrame&)>;
    void set_callback(FrameCallback cb) { callback_ = std::move(cb); }

    /// 启动解码线程（异步模式，read_one_frame 模式下不需调用）
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

    mutable std::mutex ctx_mtx_;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* dec_ctx_ = nullptr;
    AVBufferRef* hw_dev_ctx_ = nullptr;
    int video_stream_idx_ = -1;

    std::atomic<bool> running_{false};
    std::thread decode_thread_;
    int fps_ = 25;

    // 同步 read_one_frame 使用的缓冲
    AVPacket* read_pkt_ = nullptr;
    AVFrame* read_hw_frame_ = nullptr;
    AVFrame* read_sw_frame_ = nullptr;

    // 重连计数器
    int reconnect_attempts_ = 0;

    // 用于中断阻塞式 IO
    std::atomic<int64_t> last_read_time_{0};
    static int interrupt_callback(void* opaque);

    bool init_decoder();
    bool decode_one_packet(DecodedFrame* out);
    void decode_loop();
    bool reconnect();
    void cleanup();
};
