#pragma once
#include <vector>
#include <memory>
#include <atomic>
#include <functional>

#include "config.hpp"
#include "inference_engine.hpp"
#include "perf_stats.hpp"
#include "frame_pool.hpp"
#include "csrc/vision/common/image_data.h"

/// 多模型调度组：管理同一路视频上的多个推理模型
class InferGroup {
public:
    using ModelFactory = std::function<std::shared_ptr<InferenceEngine>(const ModelConfig&)>;

    explicit InferGroup(const TaskConfig& cfg,
                        ModelFactory factory = nullptr);
    ~InferGroup();

    /// 初始化所有模型（优先通过 factory 共享，否则自己加载）
    bool init();

    /// 对一帧执行所有模型的推理
    bool run_models(uint8_t* y_plane, uint8_t* uv_plane,
                    int width, int height, int y_step, int uv_step,
                    std::vector<InferResult>* results,
                    modeldeploy::vision::ImageData* frame_out = nullptr);

    PerfStats& stats() { return stats_; }
    bool ready() const;

    bool add_model(const ModelConfig& cfg);
    bool remove_model(const std::string& name);
    bool update_model(const std::string& name, const ModelConfig& cfg);

private:
    TaskConfig cfg_;
    ModelFactory factory_;
    std::vector<std::shared_ptr<InferenceEngine>> engines_;
    std::vector<int> frame_counters_;
    PerfStats stats_;
    FramePool frame_pool_;
    std::atomic<bool> initialized_{false};

    bool should_process(size_t idx);
};
