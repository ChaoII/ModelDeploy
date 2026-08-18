#include "batch_scheduler.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef WITH_GPU
#include "csrc/vision/processors/cuda/nv12_to_bgr.cuh"
#endif
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy;
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
#ifdef WITH_GPU
    if (cudaStreamCreate(&nv12_stream_) != cudaSuccess) {
        nv12_stream_ = nullptr;
    }
#endif
    sched_thread_ = std::thread(&BatchScheduler::scheduler_loop, this);
    std::cout << "[BatchScheduler] Started (batch_size=" << max_batch_size_
              << " timeout=" << batch_timeout_ms_ << "ms)" << std::endl;
    return true;
}

void BatchScheduler::stop() {
    running_ = false;
    req_cv_.notify_all();
    if (sched_thread_.joinable()) sched_thread_.join();
#ifdef WITH_GPU
    if (nv12_stream_) {
        cudaStreamDestroy(nv12_stream_);
        nv12_stream_ = nullptr;
    }
#endif
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
            auto t0 = std::chrono::steady_clock::now();
            process_batch(batch);
            auto t1 = std::chrono::steady_clock::now();
            total_batch_process_us_.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
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
    // 全 NV12 通路：设备请求零拷贝直用设备指针；host 请求紧凑 host NV12。
    // 统一 batch_predict → yolo_preprocess_nv12_batch_cuda（逐图判定设备/host，设备零 PCIe，
    // host 聚合 H2D），消除原先 NV12→BGR→back 的双重转换。
    std::vector<ImageData> nv12_images;
    nv12_images.reserve(batch.size());
    std::vector<bool> is_device(batch.size(), false);
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& [req, res] = batch[i];
        if (req.y_device && req.uv_device && req.gpu_nv12) {
            ImageData::Plane pl[2] = {{
                                          const_cast<uint8_t*>(req.y_device), req.y_step_device},
                                      {const_cast<uint8_t*>(req.uv_device), req.uv_step_device}};
            auto img = ImageData::from_planes(pl, 2, MdImageType::NV12, req.width, req.height,
                                              Device::GPU, req.gpu_nv12);
            nv12_images.push_back(std::move(img));
            is_device[i] = true;
        } else if (req.y_plane && req.uv_plane) {
            ImageData::Plane pl[2] = {{req.y_plane, req.width}, {req.uv_plane, req.width}};
            auto img = ImageData::from_planes(pl, 2, MdImageType::NV12, req.width, req.height,
                                              Device::CPU, {});
            nv12_images.push_back(std::move(img));
        } else {
            nv12_images.emplace_back();  // 空：本帧无法预处理（保位）
        }
    }

    std::lock_guard<std::mutex> lock(models_mtx_);

    for (auto& [key, entry] : models_) {
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
        bool subset_uniform = true;
        {
            const BatchRequest* ref = nullptr;
            for (size_t k : want) {
                auto& [req, res] = batch[k];
                if (!ref) { ref = &req; }
                else if (req.width != ref->width || req.height != ref->height) {
                    subset_uniform = false; break;
                }
            }
            for (size_t k : want) if (nv12_images[k].empty()) { subset_uniform = false; break; }
        }

        if (entry.prototype->config().type == "detection" && subset_uniform) {
            std::vector<ImageData> sub_images;
            sub_images.reserve(want.size());
            for (size_t k : want) sub_images.push_back(nv12_images[k]);
            auto* det = entry.prototype->det_model();
            if (det && det->is_initialized()) {
                std::vector<std::vector<DetectionResult>> all_results;
                auto it0 = std::chrono::steady_clock::now();
                const bool ok = det->batch_predict(sub_images, &all_results);
                auto it1 = std::chrono::steady_clock::now();
                total_infer_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(it1 - it0).count());
                if (ok) {
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

        // Fallback: sequential per-frame inference (NV12)
        for (size_t k : want) {
            auto& [req, res] = batch[k];
            if (nv12_images[k].empty()) continue;
            InferResult result;
            if (entry.prototype->infer(nv12_images[k], &result)) {
                if (!result.boxes.empty()) {
                    res->results.push_back(std::move(result));
                }
            }
        }
    }

    // Fill results：预览路径把（绘制后的）设备 NV12 回传，非预览路省去
    for (size_t i = 0; i < batch.size(); ++i) {
        auto& [req, res] = batch[i];
        if (req.need_nv12 && req.gpu_nv12 && is_device[i]) {
            res->nv12_gpu = req.gpu_nv12;
            res->width = req.width;
            res->height = req.height;
        }
        res->ready = true;
    }
}
