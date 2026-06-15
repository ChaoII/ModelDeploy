#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <future>

using namespace modeldeploy::vision;

InferGroup::InferGroup(const TaskConfig& cfg, ModelFactory factory)
    : cfg_(cfg), factory_(factory), frame_pool_(32) {
}

InferGroup::~InferGroup() {
    engines_.clear();
}

bool InferGroup::init() {
    for (const auto& mcfg : cfg_.models) {
        std::shared_ptr<InferenceEngine> engine;
        if (factory_) {
            // 通过模型注册表获取（共享显存）
            engine = factory_(mcfg);
        } else {
            // 没有注册表时自己加载
            auto e = std::make_shared<InferenceEngine>();
            if (!e->load(mcfg)) {
                std::cerr << "[InferGroup] Failed to load model: " << mcfg.name << std::endl;
                return false;
            }
            engine = e;
        }
        if (!engine || !engine->is_loaded()) {
            std::cerr << "[InferGroup] Model not loaded: " << mcfg.name << std::endl;
            return false;
        }
        engines_.push_back(std::move(engine));
        frame_counters_.push_back(0);
    }
    initialized_ = !engines_.empty();
    return initialized_;
}

bool InferGroup::ready() const {
    return initialized_.load();
}

bool InferGroup::should_process(size_t idx) {
    return (++frame_counters_[idx]) % engines_[idx]->config().interval == 0;
}

bool InferGroup::run_models(uint8_t* y_plane, uint8_t* uv_plane,
                             int width, int height, int y_step, int uv_step,
                             std::vector<InferResult>* results,
                             ImageData* frame_out) {
    if (!initialized_) return false;
    results->clear();

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
        *frame_out = bgr_image.clone();

    // ── 多模型并行推理（GPU 可同时执行不同 CUDA Session） ──
    struct ModelTask {
        size_t index;
        ImageData input;
        InferResult result;
        int64_t dt_us = 0;
    };
    std::vector<ModelTask> tasks;
    tasks.reserve(engines_.size());

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
        tasks.push_back({i, std::move(input_image), InferResult{}, 0});
    }

    // 同时发射所有推理任务 → GPU 并发执行 CUDA kernel
    {
        std::vector<std::future<void>> futures;
        futures.reserve(tasks.size());
        for (auto& task : tasks) {
            auto& engine = engines_[task.index];
            futures.push_back(std::async(std::launch::async, [&engine, &task]() {
                auto t0 = std::chrono::steady_clock::now();
                engine->infer(task.input, &task.result);
                task.dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            }));
        }
        for (auto& f : futures) f.wait();
    }

    for (auto& task : tasks) {
        if (!task.result.boxes.empty())
            results->push_back(std::move(task.result));
        stats_.record_frame(0, task.dt_us, 0, 0);
    }

    return !results->empty();
}

bool InferGroup::add_model(const ModelConfig& mcfg) {
    std::shared_ptr<InferenceEngine> engine;
    if (factory_) {
        engine = factory_(mcfg);
    } else {
        auto e = std::make_shared<InferenceEngine>();
        if (!e->load(mcfg)) return false;
        engine = e;
    }
    if (!engine || !engine->is_loaded()) return false;
    engines_.push_back(std::move(engine));
    frame_counters_.push_back(0);
    return true;
}

bool InferGroup::remove_model(const std::string& name) {
    for (size_t i = 0; i < engines_.size(); ++i) {
        if (engines_[i]->config().name == name) {
            engines_.erase(engines_.begin() + i);
            frame_counters_.erase(frame_counters_.begin() + i);
            return true;
        }
    }
    return false;
}

bool InferGroup::update_model(const std::string& name, const ModelConfig& mcfg) {
    for (auto& eng : engines_) {
        if (eng->config().name == name) {
            // 重新创建：让 factory 决定是否复用
            std::shared_ptr<InferenceEngine> new_eng;
            if (factory_) {
                new_eng = factory_(mcfg);
            } else {
                auto e = std::make_shared<InferenceEngine>();
                e->load(mcfg);
                new_eng = e;
            }
            if (new_eng && new_eng->is_loaded()) {
                eng = std::move(new_eng);
                return true;
            }
            return false;
        }
    }
    return false;
}
