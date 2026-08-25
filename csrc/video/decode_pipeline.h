#pragma once
#include "csrc/video/backend/decoder_backend.h"
#include "csrc/video/video_codec_config.h"
#include "csrc/video/video_common.h"
#include "csrc/video/video_frame.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace modeldeploy::video {

// 解码运行时/路由层：包装任意 DecoderBackend（FFmpeg/GStreamer 同构、不感知具体后端），
// 统一提供三类工程化能力：
//   1) 有界背压异步队列：解码线程抽帧入队，满时按 Backpressure{Block,Drop,OverwriteOldest} 处理；
//   2) 帧缓冲池：VideoFrame 容器复用（交付回调后回收到池供下一帧复用）；
//   3) 重连状态机：瞬态失败（非 EOF、非永久失败）按 reconnect_delay_ms 重试至 max_reconnects。
// 同步 read_one_frame 转发后端（不经队列），保持后端现有语义。
class MODELDEPLOY_CXX_EXPORT DecodePipeline {
public:
    using FrameCallback = std::function<void(VideoFrame&&)>;

    explicit DecodePipeline(std::shared_ptr<DecoderBackend> backend,
                            const VideoDecoderConfig& cfg = VideoDecoderConfig{});
    ~DecodePipeline();

    DecodePipeline(const DecodePipeline&) = delete;
    DecodePipeline& operator=(const DecodePipeline&) = delete;

    // 打开后端并记录当前 URL（重连需重开同一地址）；成功返回 true
    bool open(const std::string& url, std::string* err = nullptr);
    // 同步抽帧：直接转发后端（不经异步队列、不做重连）。失败/EOF 返回 false
    bool read_one_frame(VideoFrame* out, std::string* err = nullptr);
    // 异步逐帧回调（队列消费端）；帧以移动语义交付，回调返回后回收容器回池
    void set_callback(FrameCallback cb);
    // 启动解码线程 + 交付线程；返回 false（如未 open）时置 err
    bool start(std::string* err = nullptr);
    // 停止：请求线程停止、排空队列、join 两个线程。幂等
    void stop();
    void set_device_only(bool v);

    State state() const;
    const VideoStats& stats() const;
    std::string last_error() const;
    int fps() const;
    int width() const;
    int height() const;
    void close();  // 幂等：stop + 关后端 + Closed

    // 测试可观测：缓冲池命中（复用成功）次数；dropped/reconnect_count 经 stats() 读取
    uint64_t pool_hits() const;
    uint64_t pool_returns() const;
private:
    void decode_loop();
    void deliver_loop();
    // 从池取用一帧容器（pooling=false 时每帧新建）；返回非空 shared_ptr
    std::shared_ptr<VideoFrame> acquire_frame();
    // 把帧容器回收到池（pooling=false 时直接释放）
    void release_frame(std::shared_ptr<VideoFrame> f);
    // 把已解码帧推入有界队列；返回是否被接收（Drop 满时可为 false，调用方回收该帧）
    bool push_frame(std::shared_ptr<VideoFrame> f);
    void drain_queue_locked();

    std::shared_ptr<DecoderBackend> backend_;
    VideoDecoderConfig cfg_;
    std::string url_;

    // 帧缓冲池：空闲容器栈 + 互斥（mutable 以支持 const 计数读取）
    mutable std::mutex pool_mtx_;
    std::deque<std::shared_ptr<VideoFrame>> pool_free_;
    bool pooling_ = true;
    uint64_t pool_hits_ = 0;
    uint64_t pool_returns_ = 0;

    // 有界异步队列 + 条件变量
    std::mutex qmtx_;
    std::condition_variable qcond_not_empty_;
    std::condition_variable qcond_not_full_;
    std::deque<std::shared_ptr<VideoFrame>> queue_;

    FrameCallback callback_;
    std::thread decode_thread_;
    std::thread deliver_thread_;
    std::atomic<bool> stop_requested_{false};
    bool started_ = false;

    // 解码会话状态：原子化以支持测试/使用方跨线程安全轮询
    std::atomic<int> state_{static_cast<int>(State::Idle)};
    std::string err_;
    VideoStats stats_;
};

} // namespace modeldeploy::video
