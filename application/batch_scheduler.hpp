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

/// Batch request: one pipeline submits a frame for batched inference
struct BatchRequest {
    std::string pipeline_id;
    uint8_t* y_plane = nullptr;
    uint8_t* uv_plane = nullptr;
    int width = 0;
    int height = 0;
};

/// Batch result: returned to the pipeline after inference
struct BatchResult {
    std::string pipeline_id;
    std::vector<InferResult> results;
    modeldeploy::vision::ImageData bgr_image;
    int64_t infer_us = 0;
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
    std::vector<uint8_t> bgr_buf_;
    int last_w_ = 0, last_h_ = 0;

    PerfStats stats_;

    void scheduler_loop();
    void process_batch(std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>>& batch);
};