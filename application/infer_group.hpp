#pragma once
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "config.hpp"
#include "inference_engine.hpp"
#include "perf_stats.hpp"
#include "frame_pool.hpp"
#include "csrc/vision/common/image_data.h"

/// 多模型调度组：管理同一路视频上的多个推理模型
/// 每模型一个常驻 worker 线程，避免 std::async 反复创建线程
class InferGroup {
public:
    explicit InferGroup(const TaskConfig& cfg);
    ~InferGroup();

    /// 初始化所有模型 + 启动 worker 线程池
    bool init();

    /// 对一帧执行所有模型的推理（多模型并行）
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
    std::vector<std::unique_ptr<InferenceEngine>> engines_;
    std::vector<int> frame_counters_;
    PerfStats stats_;
    FramePool frame_pool_;
    std::atomic<bool> initialized_{false};

    // 每个模型一个常驻 worker（避免 std::async 重复创建线程）
    struct Worker {
        std::thread thread;
        std::mutex mtx;
        std::condition_variable cv_in;
        std::condition_variable cv_out;
        std::function<void()> task;
        bool has_task = false;
        bool done = true;
        bool stop = false;
    };
    std::vector<std::unique_ptr<Worker>> workers_;

    bool should_process(size_t idx);
    void worker_loop(Worker* w);
    void start_workers();
    void stop_workers();
};
