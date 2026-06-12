#pragma once
#include <vector>
#include <memory>
#include <atomic>

#include "config.hpp"
#include "inference_engine.hpp"
#include "perf_stats.hpp"
#include "frame_pool.hpp"

/// 多模型调度组：管理同一路视频上的多个推理模型
class InferGroup {
public:
    explicit InferGroup(const TaskConfig& cfg);
    ~InferGroup();

    /// 初始化所有模型
    bool init();

    /// 对一帧执行所有模型的推理
    bool run_models(uint8_t* y_plane, uint8_t* uv_plane,
                    int width, int height, int y_step, int uv_step,
                    std::vector<InferResult>* results);

    /// 获取性能统计
    PerfStats& stats() { return stats_; }

    /// 是否所有模型就绪
    bool ready() const;

    /// 动态操作模型
    bool add_model(const ModelConfig& cfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& cfg);

private:
    TaskConfig cfg_;
    std::vector<std::unique_ptr<InferenceEngine>> engines_;
    std::vector<int> frame_counters_;     // 每个模型的帧计数器（用于跳帧）
    PerfStats stats_;
    FramePool frame_pool_;
    std::atomic<bool> initialized_{false};

    /// 检查是否需要处理当前帧（基于 interval 跳帧）
    bool should_process(size_t idx);

};
