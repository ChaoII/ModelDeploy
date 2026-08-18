#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <string>

#include "config.hpp"
#include "inference_engine.hpp"
#include "perf_stats.hpp"
#include "csrc/vision/common/image_data.h"
#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

/// Batch request: one pipeline submits a frame for batched inference
struct BatchRequest {
    std::string pipeline_id;
    std::vector<std::string> model_names;  // models this pipeline needs for this frame (empty = all)
    // host NV12（紧凑，step==width）；设备直通时以 y_device/uv_device 优先
    uint8_t* y_plane = nullptr;
    uint8_t* uv_plane = nullptr;
    int width = 0;
    int height = 0;
    // CUVID 硬解设备 NV12 指针（GPU 直通零拷贝）
    const uint8_t* y_device = nullptr;
    const uint8_t* uv_device = nullptr;
    int y_step_device = 0;
    int uv_step_device = 0;
    // device NV12 缓冲所有权（池块），跨批生命周期安全；host 请求可为空
    std::shared_ptr<uint8_t> gpu_nv12;
    bool need_nv12 = false;  // 是否需要把（绘制后的）NV12 帧回传（预览路径；非预览路省去）
};

/// Batch result: returned to the pipeline after inference
struct BatchResult {
    std::string pipeline_id;
    std::vector<InferResult> results;
    modeldeploy::vision::ImageData bgr_image;
    // 设备 NV12 输出（含绘制），预览路径；非预览路为空
    std::shared_ptr<uint8_t> nv12_gpu;
    int width = 0;
    int height = 0;
    int64_t us = 0;       // submit→ready 等待耗时（us）
    bool ready = false;
};

/// BatchScheduler: collects frames from multiple pipelines and runs
/// batched inference on models that share the same weights.
///
/// How it works:
/// 1. Pipelines submit BatchRequest via submit()
/// 2. Returns a shared_ptr<BatchResult> that will be filled when ready
/// 3. A scheduler thread collects pending requests into batches
/// 4. For each unique model, runs batched inference (or falls back to sequential)
/// 5. Distributes results back to pipelines
///
/// Batch inference requires all frames in a batch to have the same model input size.
class BatchScheduler {
public:
    explicit BatchScheduler(int max_batch_size = 4,
                            int batch_timeout_ms = 10);
    ~BatchScheduler();

    /// Submit a frame for batched inference. Returns a future-like result.
    std::shared_ptr<BatchResult> submit(const BatchRequest& req);

    /// Register a model (prototype) that will be shared across pipelines
    bool register_model(const ModelConfig& cfg);

    /// Start the scheduler thread
    bool start();

    /// Stop the scheduler thread
    void stop();

    bool is_running() const { return running_.load(); }

    const PerfStats& stats() const { return stats_; }

    /// Average observed batch size (verification metric for batching)
    double avg_batch_size() const {
        uint64_t n = total_batches_.load();
        return n ? static_cast<double>(total_batched_frames_.load()) / n : 0.0;
    }

    /// Average process_batch wall time in microseconds
    double avg_batch_process_us() const {
        uint64_t n = total_batches_.load();
        return n ? static_cast<double>(total_batch_process_us_.load()) / n : 0.0;
    }

    /// Average batch_predict (在 process_batch 内部) 耗时 in microseconds
    double avg_batch_infer_us() const {
        uint64_t n = total_batches_.load();
        return n ? static_cast<double>(total_infer_us_.load()) / n : 0.0;
    }

private:
    int max_batch_size_;
    int batch_timeout_ms_;
    std::atomic<bool> running_{false};
    std::atomic<bool> started_{false};
    std::thread sched_thread_;

    // Pending requests (guarded by req_mtx_)
    std::mutex req_mtx_;
    std::condition_variable req_cv_;
    std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>> pending_;

    // Model cache: model cache key → prototype engine
    struct ModelEntry {
        ModelConfig cfg;
        std::unique_ptr<InferenceEngine> prototype;
    };
    std::unordered_map<std::string, ModelEntry> models_;
    std::mutex models_mtx_;

    // NV12 buf for BGR conversion (reused across batches)
    std::vector<uint8_t> nv12_buf_;
    int last_w_ = 0, last_h_ = 0;

#ifdef WITH_GPU
    cudaStream_t nv12_stream_ = nullptr;  // 复用 CUDA 流，避免每批 create/destroy + 隐式同步
#endif

    PerfStats stats_;

    // Batch statistics
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_batched_frames_{0};
    std::atomic<uint64_t> total_batch_process_us_{0};
    std::atomic<uint64_t> total_infer_us_{0};

    void scheduler_loop();
    void process_batch(std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>>& batch);
};