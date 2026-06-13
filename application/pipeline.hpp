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
#include "infer_group.hpp"
#include "draw_engine.hpp"
#include "stream_encoder.hpp"

/// 待处理帧（解码线程→流水线线程）
struct PendingFrame {
    std::vector<uint8_t> nv12_data;
    int width = 0, height = 0;
};

/// 单路视频流水线：decoder → infer_group → draw → encoder
class Pipeline {
public:
    explicit Pipeline(TaskConfig cfg);
    ~Pipeline();

    /// 启动流水线（返回后流水线在后台初始化）
    bool start();

    /// 停止流水线
    void stop();

    /// 是否正在运行
    bool is_running() const { return running_.load(); }

    /// 是否已成功初始化
    bool is_initialized() const { return initialized_.load(); }

    const std::string& task_id() const { return cfg_.id; }
    const PerfStats& stats() const { return stats_; }
    const TaskConfig& config() const { return cfg_; }

    bool add_model(const ModelConfig& mcfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& mcfg);

private:
    TaskConfig cfg_;
    std::unique_ptr<StreamDecoder> decoder_;
    std::unique_ptr<InferGroup> infer_group_;
    std::unique_ptr<DrawEngine> draw_engine_;
    std::unique_ptr<StreamEncoder> encoder_;
    PerfStats stats_;
    std::atomic<bool> encoder_opened_{false};
    std::atomic<bool> initialized_{false};
    std::string init_error_;

    std::atomic<bool> running_{false};
    std::thread pipeline_thread_;

    // 帧队列（解码线程→流水线线程）
    std::queue<PendingFrame> frame_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_ = 10;

    /// 解码帧回调（解码线程调用）
    bool on_decoded_frame(const DecodedFrame& frame);

    /// 流水线主循环
    void pipeline_loop();

    /// 处理一帧（在流水线线程中）
    bool process_frame(const uint8_t* nv12, int width, int height);
};
