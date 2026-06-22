#include "batch_scheduler.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef WITH_GPU
#include "csrc/vision/common/processors/nv12_to_bgr.cuh"
#endif
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy::vision;

BatchScheduler::BatchScheduler(int max_batch_size, int batch_timeout_ms)
    : max_batch_size_(max_batch_size), batch_timeout_ms_(batch_timeout_ms) {}

BatchScheduler::~BatchScheduler() {
    stop();
}

bool BatchScheduler::register_model(const ModelConfig& cfg) {
    std::lock_guard<std::mutex> lock(models_mtx_);
    std::string key = InferenceEngine::make_cache_key(cfg);
    if (models_.count(key)) return true;

    ModelEntry entry;
    entry.cfg = cfg;
    entry.prototype = std::make_unique<InferenceEngine>();
    if (!entry.prototype->load(cfg)) {
        std::cerr << "[BatchScheduler] Failed to load model: " << cfg.name << std::endl;
        return false;
    }
    models_[key] = std::move(entry);
    std::cout << "[BatchScheduler] Registered model: " << cfg.name
              << " key=" << key << std::endl;
    return true;
}

std::shared_ptr<BatchResult> BatchScheduler::submit(const BatchRequest& req) {
    auto result = std::make_shared<BatchResult>();
    result->pipeline_id = req.pipeline_id;

    {
        std::lock_guard<std::mutex> lock(req_mtx_);
        pending_.emplace_back(req, result);
    }
    req_cv_.notify_one();
    return result;
}

bool BatchScheduler::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return true;
    running_ = true;
    sched_thread_ = std::thread(&BatchScheduler::scheduler_loop, this);
    std::cout << "[BatchScheduler] Started (batch_size=" << max_batch_size_
              << " timeout=" << batch_timeout_ms_ << "ms)" << std::endl;
    return true;
}

void BatchScheduler::stop() {
    running_ = false;
    req_cv_.notify_all();
    if (sched_thread_.joinable()) sched_thread_.join();
    started_ = false;
}

void BatchScheduler::scheduler_loop() {
    while (running_.load()) {
        std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>> batch;

        {
            std::unique_lock<std::mutex> lock(req_mtx_);
            req_cv_.wait_for(lock, std::chrono::milliseconds(batch_timeout_ms_),
                [this]() { return !pending_.empty() || !running_.load(); });

            if (!running_.load() && pending_.empty()) break;

            int count = std::min(static_cast<int>(pending_.size()), max_batch_size_);
            if (count > 0) {
                batch.assign(pending_.begin(), pending_.begin() + count);
                pending_.erase(pending_.begin(), pending_.begin() + count);
            }
        }

        if (!batch.empty()) {
            process_batch(batch);
        }
    }

    std::lock_guard<std::mutex> lock(req_mtx_);
    for (auto& [req, res] : pending_) {
        res->ready = true;
    }
    pending_.clear();
}

void BatchScheduler::process_batch(
    std::vector<std::pair<BatchRequest, std::shared_ptr<BatchResult>>>& batch) {
    // P3.2: True batch inference via batch_predict
    // Preprocess all frames to BGR
    std::vector<ImageData> bgr_images;
    bgr_images.reserve(batch.size());

    for (auto& [req, res] : batch) {
        const size_t y_size = static_cast<size_t>(req.height) * req.width;
        const size_t uv_size = y_size / 2;

        if (last_w_ != req.width || last_h_ != req.height || nv12_buf_.size() != y_size + uv_size) {
            nv12_buf_.resize(y_size + uv_size);
            bgr_buf_.resize(req.width * req.height * 3);
            last_w_ = req.width;
            last_h_ = req.height;
        }

        std::memcpy(nv12_buf_.data(), req.y_plane, y_size);
        std::memcpy(nv12_buf_.data() + y_size, req.uv_plane, uv_size);

#ifdef WITH_GPU
        nv12_to_bgr_cuda(nv12_buf_.data(), nv12_buf_.data() + y_size,
                          req.width, req.height, req.width, req.width,
                          bgr_buf_.data());
        auto bgr_image = ImageData::from_raw(bgr_buf_.data(), req.width, req.height,
                                               MdImageType::PKG_BGR_U8, false);
#else
        auto nv12_image = ImageData::from_raw(nv12_buf_.data(), req.width, req.height,
                                                MdImageType::NV12, true);
        auto bgr_image = ImageData::cvt_color(nv12_image, ColorConvertType::CVT_NV122PA_BGR);
#endif
        bgr_images.push_back(std::move(bgr_image));
    }

    // 检查 batch 内所有帧尺寸是否一致；不一致则回退逐帧
    bool uniform_size = true;
    int ref_w = batch[0].first.width, ref_h = batch[0].first.height;
    for (size_t i = 1; i < batch.size(); ++i) {
        if (batch[i].first.width != ref_w || batch[i].first.height != ref_h) {
            uniform_size = false;
            break;
        }
    }

    std::lock_guard<std::mutex> lock(models_mtx_);

    for (auto& [key, entry] : models_) {
        if (entry.prototype->config().type == "detection" && uniform_size && batch.size() > 1) {
            // True batch inference for detection models
            auto* det = entry.prototype->det_model();
            if (det && det->is_initialized()) {
                std::vector<std::vector<DetectionResult>> all_results;
                if (det->batch_predict(bgr_images, &all_results)) {
                    for (size_t i = 0; i < batch.size() && i < all_results.size(); ++i) {
                        auto& [req, res] = batch[i];
                        for (auto& d : all_results[i]) {
                            InferResult r;
                            r.model_name = entry.cfg.name;
                            r.type = "detection";
                            DetectionBox box;
                            box.x = d.box.x; box.y = d.box.y;
                            box.w = d.box.width; box.h = d.box.height;
                            box.score = d.score;
                            box.label_id = d.label_id;
                            r.boxes.push_back(box);
                            res->results.push_back(std::move(r));
                        }
                    }
                    continue; // batch successful
                }
            }
        }

        // Fallback: sequential per-frame inference
        for (size_t i = 0; i < batch.size(); ++i) {
            auto& [req, res] = batch[i];
            InferResult result;
            if (entry.prototype->infer(bgr_images[i], &result)) {
                if (!result.boxes.empty()) {
                    res->results.push_back(std::move(result));
                }
            }
        }
    }

    // Fill results
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& [req, res] = batch[i];
        auto t0 = std::chrono::steady_clock::now();
        res->bgr_image = bgr_images[i];
        res->ready = true;
        auto t1 = std::chrono::steady_clock::now();
        res->infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
}