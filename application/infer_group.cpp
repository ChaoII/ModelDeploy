#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace modeldeploy::vision;

InferGroup::InferGroup(const TaskConfig& cfg)
    : cfg_(cfg), frame_pool_(32) {
}

InferGroup::~InferGroup() {
    stop_workers();
    engines_.clear();
}

bool InferGroup::init() {
    for (const auto& mcfg : cfg_.models) {
        auto engine = std::make_unique<InferenceEngine>();
        if (!engine->load(mcfg)) {
            std::cerr << "[InferGroup] Failed to load model: " << mcfg.name << std::endl;
            return false;
        }
        engines_.push_back(std::move(engine));
        frame_counters_.push_back(0);
    }
    if (engines_.empty()) return false;
    start_workers();
    initialized_ = true;
    return true;
}

bool InferGroup::ready() const {
    return initialized_.load();
}

bool InferGroup::should_process(size_t idx) {
    return (++frame_counters_[idx]) % engines_[idx]->config().interval == 0;
}

void InferGroup::worker_loop(Worker* w) {
    cudaSetDevice(0); // 与主线程同 CUDA context
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(w->mtx);
            w->cv_in.wait(lock, [w]() { return w->has_task || w->stop; });
            if (w->stop && !w->has_task) return;
            task = std::move(w->task);
            w->has_task = false;
        }
        try { task(); } catch (...) {}
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->done = true;
        }
        w->cv_out.notify_one();
    }
}

void InferGroup::start_workers() {
    workers_.clear();
    workers_.reserve(engines_.size());
    for (size_t i = 0; i < engines_.size(); ++i) {
        auto w = std::make_unique<Worker>();
        Worker* wp = w.get();
        w->thread = std::thread([this, wp]() { this->worker_loop(wp); });
        workers_.push_back(std::move(w));
    }
}

void InferGroup::stop_workers() {
    for (auto& w : workers_) {
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->stop = true;
        }
        w->cv_in.notify_one();
    }
    for (auto& w : workers_) {
        if (w->thread.joinable()) w->thread.join();
    }
    workers_.clear();
}

bool InferGroup::run_models(uint8_t* y_plane, uint8_t* uv_plane,
                             int width, int height, int y_step, int uv_step,
                             std::vector<InferResult>* results,
                             ImageData* frame_out) {
    if (!initialized_) return false;
    results->clear();

    // ── NV12 → BGR（CPU 一次，所有模型复用） ──
    size_t y_size = static_cast<size_t>(height) * width;
    size_t uv_size = y_size / 2;
    size_t total = y_size + uv_size;

    std::vector<uint8_t> cpu_buf(total);
    for (int row = 0; row < height; ++row)
        memcpy(cpu_buf.data() + row * width, y_plane + row * y_step, width);
    int uv_h = height / 2;
    for (int row = 0; row < uv_h; ++row)
        memcpy(cpu_buf.data() + y_size + row * width, uv_plane + row * uv_step, width);

    auto nv12_image = ImageData::from_raw(cpu_buf.data(), width, height, MdImageType::NV12, true);
    auto bgr_image = ImageData::cvt_color(nv12_image, ColorConvertType::CVT_NV122PA_BGR);

    if (frame_out)
        *frame_out = bgr_image; // ImageData 浅拷贝

    // ── 多模型并行推理（每模型独立 worker，GPU 上的 CUDA kernel 可并发执行） ──
    struct ModelTask {
        size_t index;
        ImageData input;
        InferResult result;
        int64_t dt_us = 0;
        bool used = false;
    };
    std::vector<ModelTask> tasks(engines_.size());

    for (size_t i = 0; i < engines_.size(); ++i) {
        if (!should_process(i)) continue;
        auto& engine = engines_[i];
        const auto& mcfg = engine->config();
        ImageData input_image = bgr_image;
        if (mcfg.roi[2] > 0 && mcfg.roi[3] > 0) {
            Rect2f roi_rect = {static_cast<float>(mcfg.roi[0]),
                               static_cast<float>(mcfg.roi[1]),
                               static_cast<float>(mcfg.roi[2]),
                               static_cast<float>(mcfg.roi[3])};
            input_image = bgr_image.crop(roi_rect);
            if (input_image.empty()) continue;
        }
        tasks[i].index = i;
        tasks[i].input = std::move(input_image);
        tasks[i].used = true;
    }

    // 派发到 worker 池
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!tasks[i].used) continue;
        auto& w = workers_[i];
        auto& engine = engines_[i];
        ModelTask* tp = &tasks[i];
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->done = false;
            w->has_task = true;
            w->task = [&engine, tp]() {
                auto t0 = std::chrono::steady_clock::now();
                engine->infer(tp->input, &tp->result);
                tp->dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            };
        }
        w->cv_in.notify_one();
    }

    // 等待全部完成
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!tasks[i].used) continue;
        auto& w = workers_[i];
        std::unique_lock<std::mutex> lock(w->mtx);
        w->cv_out.wait(lock, [&w]() { return w->done; });
    }

    // 收集结果
    for (auto& task : tasks) {
        if (!task.used) continue;
        if (!task.result.boxes.empty())
            results->push_back(std::move(task.result));
        stats_.record_frame(0, task.dt_us, 0, 0);
    }

    return !results->empty();
}

bool InferGroup::add_model(const ModelConfig& mcfg) {
    auto engine = std::make_unique<InferenceEngine>();
    if (!engine->load(mcfg)) return false;
    // 重启 worker 池以匹配新模型数量
    stop_workers();
    engines_.push_back(std::move(engine));
    frame_counters_.push_back(0);
    start_workers();
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    for (size_t i = 0; i < engines_.size(); ++i) {
        if (engines_[i]->config().name == name) {
            stop_workers();
            engines_.erase(engines_.begin() + i);
            frame_counters_.erase(frame_counters_.begin() + i);
            start_workers();
            return true;
        }
    }
    return false;
}

bool InferGroup::update_model(const std::string& name, const ModelConfig& mcfg) {
    for (auto& eng : engines_) {
        if (eng->config().name == name) {
            stop_workers();
            eng->unload();
            bool ok = eng->load(mcfg);
            start_workers();
            return ok;
        }
    }
    return false;
}
