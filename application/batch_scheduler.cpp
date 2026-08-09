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
            // Phase 1: wait for first request (or shutdown)
            req_cv_.wait(lock, [this]() { return !pending_.empty() || !running_.load(); });
            if (!running_.load() && pending_.empty()) break;

            // Phase 2: collect more frames for up to batch_timeout_ms_
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(batch_timeout_ms_);
            while (std::chrono::steady_clock::now() < deadline &&
                   static_cast<int>(pending_.size()) < max_batch_size_) {
                req_cv_.wait_until(lock, deadline);
                // re-check pending size each wake
            }

            int count = std::min(static_cast<int>(pending_.size()), max_batch_size_);
            if (count > 0) {
                batch.assign(pending_.begin(), pending_.begin() + count);
                pending_.erase(pending_.begin(), pending_.begin() + count);
            }
        }

        if (!batch.empty()) {
            total_batched_frames_.fetch_add(batch.size());
            total_batches_.fetch_add(1);
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
        // copy=true：bgr_buf_ 为跨批复用的成员缓冲，必须深拷贝，否则
        // bgr_image 别名同一缓冲，下一帧/下一批会覆盖已返回给 pipeline 的数据
        auto bgr_image = ImageData::from_raw(bgr_buf_.data(), req.width, req.height,
                                               MdImageType::PKG_BGR_U8, true);
#else
        auto nv12_image = ImageData::from_raw(nv12_buf_.data(), req.width, req.height,
                                                MdImageType::NV12, true);
        auto bgr_image = ImageData::cvt_color(nv12_image, ColorConvertType::CVT_NV122PA_BGR);
#endif
        bgr_images.push_back(std::move(bgr_image));
    }

    // 检查 batch 内所有帧尺寸是否一致；不一致则回退逐帧
    std::lock_guard<std::mutex> lock(models_mtx_);

    for (auto& [key, entry] : models_) {
        // 仅处理请求了本模型（或 model_names 为空 = 全部）的帧，避免把
        // 本模型结果塞给不相关的 pipeline（多模型部署正确性）。
        std::vector<size_t> want;
        want.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& req = batch[i].first;
            if (req.model_names.empty() ||
                std::find(req.model_names.begin(), req.model_names.end(),
                          entry.cfg.name) != req.model_names.end()) {
                want.push_back(i);
            }
        }
        if (want.empty()) continue;

        // Batch 推理要求该模型命中的帧尺寸一致；不一致则回退逐帧
        bool subset_uniform = want.size() > 1;
        if (subset_uniform) {
            const int ref_w = batch[want[0]].first.width;
            const int ref_h = batch[want[0]].first.height;
            for (size_t k = 1; k < want.size(); ++k) {
                if (batch[want[k]].first.width != ref_w ||
                    batch[want[k]].first.height != ref_h) {
                    subset_uniform = false;
                    break;
                }
            }
        }

        if (entry.prototype->config().type == "detection" && subset_uniform) {
            // True batch inference for detection models
            std::vector<ImageData> sub_images;
            sub_images.reserve(want.size());
            for (size_t k : want) sub_images.push_back(bgr_images[k]);
            auto* det = entry.prototype->det_model();
            if (det && det->is_initialized()) {
                std::vector<std::vector<DetectionResult>> all_results;
                if (det->batch_predict(sub_images, &all_results)) {
                    for (size_t k = 0; k < want.size() && k < all_results.size(); ++k) {
                        auto& [req, res] = batch[want[k]];
                        for (auto& d : all_results[k]) {
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
        for (size_t k : want) {
            auto& [req, res] = batch[k];
            InferResult result;
            if (entry.prototype->infer(bgr_images[k], &result)) {
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