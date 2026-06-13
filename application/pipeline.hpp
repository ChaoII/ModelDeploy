#pragma once
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

#include "config.hpp"
#include "perf_stats.hpp"
#include "stream_decoder.hpp"
#include "infer_group.hpp"
#include "draw_engine.hpp"
#include "stream_encoder.hpp"

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

    /// 是否已成功初始化（模型加载、解码器连接完成）
    bool is_initialized() const { return initialized_.load(); }

    /// 获取任务 ID
    const std::string& task_id() const { return cfg_.id; }

    /// 获取性能统计
    const PerfStats& stats() const { return stats_; }

    /// 获取任务配置
    const TaskConfig& config() const { return cfg_; }

    /// 更新模型配置（运行时）
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

    /// 解码帧回调（解码线程调用 -> 推入流水线）
    bool on_decoded_frame(const DecodedFrame& frame);

    /// 流水线主循环（在 pipeline_thread_ 中运行）
    void pipeline_loop();
};
