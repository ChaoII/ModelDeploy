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

#include "config.hpp"
#include "perf_stats.hpp"
#include "stream_decoder.hpp"
#include "stream_hub.hpp"
#include "infer_group.hpp"
#include "draw_engine.hpp"
#include "stream_encoder.hpp"

namespace modeldeploy { namespace vision { class ImageData; } }

/// 待处理帧（解码线程→流水线线程）
/// 优先持有 BroadcastFrame 引用避免拷贝；若来自独占解码器则用 nv12_data
struct PendingFrame {
    std::shared_ptr<BroadcastFrame> shared_frame; // 来自 StreamHub（零拷贝）
    std::vector<uint8_t> nv12_data;               // 独占模式才用
    int width = 0, height = 0;

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

/// 单路视频流水线：decoder → infer_group → draw → encoder
class Pipeline {
public:
    /// 不传 hub 时自己创建 StreamDecoder（独占模式）
    /// 传 hub 时通过 hub 订阅共享解码源（推荐）
    explicit Pipeline(TaskConfig cfg, StreamHub* hub = nullptr);
    ~Pipeline();

    /// 启动流水线（返回后流水线在后台初始化）
    bool start();

    /// 停止流水线
    void stop();

    /// 是否正在运行
    bool is_running() const { return running_.load(); }

    /// 是否已成功初始化
    bool is_initialized() const { return initialized_.load(); }

    /// 初始化失败原因
    const std::string& init_error() const { return init_error_; }

    const std::string& task_id() const { return cfg_.id; }
    const PerfStats& stats() const { return stats_; }
    const TaskConfig& config() const { return cfg_; }

    bool add_model(const ModelConfig& mcfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& mcfg);

    /// 获取最新一帧 BGR JPEG 编码（线程安全）
    bool latest_jpeg(std::vector<uint8_t>* out, int quality = 80);

private:
    TaskConfig cfg_;
    StreamHub* hub_ = nullptr;                    // 流共享中心（可选）
    std::shared_ptr<SharedSource> shared_source_; // 当前订阅的源
    uint64_t shared_token_ = 0;                   // 订阅 token
    std::unique_ptr<StreamDecoder> decoder_;      // 独占模式才用
    std::unique_ptr<InferGroup> infer_group_;
    std::unique_ptr<DrawEngine> draw_engine_;
    std::unique_ptr<StreamEncoder> encoder_;
    PerfStats stats_;
    std::atomic<bool> encoder_opened_{false};
    std::atomic<bool> initialized_{false};
    std::string init_error_;

    // 最新一帧（ImageData 浅拷贝，shared_ptr 内部管理生命周期）
    std::mutex snapshot_mtx_;
    std::shared_ptr<modeldeploy::vision::ImageData> latest_bgr_;

    std::atomic<bool> running_{false};
    std::thread pipeline_thread_;

    // 帧队列（解码线程→流水线线程）
    std::queue<PendingFrame> frame_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::condition_variable queue_not_full_cv_;
    size_t max_queue_size_ = 3;
    bool block_on_queue_full_ = false; // 文件输入时阻塞，避免丢帧

    /// 计算合理的帧队列大小（按模型数量自适应）
    size_t calc_queue_size() const;

    /// 解码帧回调（解码线程调用，独占模式）
    bool on_decoded_frame(const DecodedFrame& frame);

    /// 共享源帧回调（StreamHub 模式）
    bool on_shared_frame(const std::shared_ptr<BroadcastFrame>& frame);

    /// 流水线主循环
    void pipeline_loop();

    /// 处理一帧（在流水线线程中）
    bool process_frame(const uint8_t* nv12, int width, int height);

    /// 安全释放所有资源（仅在流水线线程或停止后调用）
    void release_resources();
};
