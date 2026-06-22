#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>

#include "config.hpp"
#include "perf_stats.hpp"
#include "stream_decoder.hpp"
#include "stream_hub.hpp"
#include "infer_group.hpp"
#include "draw_engine.hpp"
#include "stream_encoder.hpp"
#include "batch_scheduler.hpp"

namespace modeldeploy { namespace vision { class ImageData; } }

/// 解码段→推理段 的待处理帧（持有 NV12 数据所有权或共享指针）
struct PendingFrame {
    std::shared_ptr<BroadcastFrame> shared_frame; // StreamHub 共享解码器
    std::vector<uint8_t> nv12_data;               // 独占解码器：自有 NV12 缓冲
    int width = 0, height = 0;
    int64_t pts = 0;
    double wall_time_sec = 0.0; // 帧到达解码段的墙钟时刻（用于实时丢帧判断）

    const uint8_t* y_ptr() const {
        if (shared_frame) return shared_frame->nv12_data.data();
        return nv12_data.data();
    }
    const uint8_t* uv_ptr() const {
        size_t y_size = static_cast<size_t>(height) * width;
        if (shared_frame) return shared_frame->nv12_data.data() + y_size;
        return nv12_data.data() + y_size;
    }
};

/// 推理段→编码段 的已处理帧（已绘制 BGR ImageData）
struct EncodedFrame {
    modeldeploy::vision::ImageData bgr_image;
    int width = 0, height = 0;
    int64_t pts = 0;
};

/// 单路视频流水线：三段异步流水线
/// 解码线程 (decode) ──[in_queue]──> 推理+绘制线程 (process) ──[out_queue]──> 编码线程 (encode)
/// 各段独立速率，慢端通过队列丢老帧实现背压，互不拖累
class Pipeline {
public:
    using ModelFactory = std::function<std::unique_ptr<InferenceEngine>(const ModelConfig&)>;

    explicit Pipeline(TaskConfig cfg, StreamHub* hub = nullptr,
                      ModelFactory model_factory = nullptr,
                      BatchScheduler* batch_scheduler = nullptr);
    ~Pipeline();

    bool start();
    void stop();
    bool is_running() const { return running_.load(); }
    bool is_initialized() const { return initialized_.load(); }
    const std::string& init_error() const { return init_error_; }

    const std::string& task_id() const { return cfg_.id; }
    const PerfStats& stats() const { return stats_; }
    const TaskConfig& config() const { return cfg_; }

    bool add_model(const ModelConfig& mcfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& mcfg);

    /// 更新任务核心配置（仅支持 enable_preview、encoder、decoder、draw 字段；需停止任务后调用）
    bool update_config(const TaskConfig& cfg);
    void update_preview_mode(bool enable);

    /// 获取最新一帧 BGR JPEG（线程安全）
    bool latest_jpeg(std::vector<uint8_t>* out, int quality = 80);

    /// 是否启用预览编码
    bool is_preview_enabled() const { return cfg_.enable_preview; }

private:
    TaskConfig cfg_;
    StreamHub* hub_ = nullptr;
    ModelFactory model_factory_;
    BatchScheduler* batch_scheduler_ = nullptr;
    std::shared_ptr<SharedSource> shared_source_;
    uint64_t shared_token_ = 0;
    std::unique_ptr<StreamDecoder> decoder_;
    std::unique_ptr<InferGroup> infer_group_;
    std::unique_ptr<DrawEngine> draw_engine_;
    std::unique_ptr<StreamEncoder> encoder_;
    PerfStats stats_;
    std::atomic<bool> encoder_opened_{false};
    std::atomic<bool> initialized_{false};
    std::string init_error_;

    // 最新一帧（用于 HTTP 快照）
    std::mutex snapshot_mtx_;
    std::shared_ptr<modeldeploy::vision::ImageData> latest_bgr_;
    int snapshot_interval_ = 2;

    // 最新检测框缓存：跳推理帧复用上次结果绘制，保持源帧率编码画面有框
    std::vector<InferResult> cached_results_;

    // 源流帧率（从解码器自动检测）
    int source_fps_ = 25;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopped_{true};       // 防止重复 stop
    std::thread decode_thread_;
    std::thread process_thread_;
    std::thread encode_thread_;

    // ── 解码段 → 推理段 ──
    std::queue<PendingFrame> in_queue_;
    std::mutex in_mtx_;
    std::condition_variable in_cv_;
    size_t in_max_size_ = 3;
    bool block_on_in_full_ = false;

    // ── 推理段 → 编码段 ──
    std::queue<EncodedFrame> out_queue_;
    std::mutex out_mtx_;
    std::condition_variable out_cv_;
    size_t out_max_size_ = 3;

    // ── 解码段统计 ──
    std::atomic<int64_t> last_decode_us_{0};
    std::atomic<int64_t> last_encode_us_{0};
    std::atomic<int64_t> last_frame_pts_{0};

    size_t calc_in_queue_size() const;

    /// 解码回调（StreamHub 模式）
    bool on_shared_frame(const std::shared_ptr<BroadcastFrame>& frame);

    /// 三段循环
    void decode_loop();
    void process_loop();
    void encode_loop();

    /// 推送一帧到 in_queue_（解码段调用）
    void push_in_queue(PendingFrame&& pf);

    /// 资源安全释放（仅在所有线程结束后调用）
    void release_resources();
};
