#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace modeldeploy::vision;

InferGroup::InferGroup(const TaskConfig& cfg)
    : cfg_(cfg), frame_pool_(32) {
}

InferGroup::~InferGroup() {
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
    initialized_ = engines_.empty() ? false : true;
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
                             std::vector<InferResult>* results) {
    if (!initialized_) return false;
    results->clear();

    // 从 GPU 下载 NV12 到 CPU 连续 buffer
    size_t y_size = static_cast<size_t>(height) * width;
    size_t uv_size = y_size / 2;
    size_t total = y_size + uv_size;

    auto* cpu_buf = frame_pool_.acquire(total);
    if (!cpu_buf) return false;

    cudaMemcpy2D(cpu_buf, width, y_plane, y_step, width, height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(cpu_buf + y_size, width, uv_plane, uv_step, width, height / 2, cudaMemcpyDeviceToHost);

    // 创建 NV12 ImageData
    auto nv12_image = ImageData::from_raw(cpu_buf, width, height, MdImageType::NV12, true);
    frame_pool_.release(cpu_buf);

    if (nv12_image.empty()) return false;

    // 对每个模型执行推理
    for (size_t i = 0; i < engines_.size(); ++i) {
        if (!should_process(i)) continue;

        auto& engine = engines_[i];
        const auto& mcfg = engine->config();

        // ROI 裁剪
        ImageData input_image = nv12_image;
        bool has_roi = (mcfg.roi[2] > 0 && mcfg.roi[3] > 0);
        if (has_roi) {
            // crop() expects Rect2f with float coordinates
            Rect2f roi_rect = {static_cast<float>(mcfg.roi[0]),
                               static_cast<float>(mcfg.roi[1]),
                               static_cast<float>(mcfg.roi[2]),
                               static_cast<float>(mcfg.roi[3])};
            input_image = nv12_image.crop(roi_rect);
            if (input_image.empty()) continue;
        }

        InferResult result;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = engine->infer(input_image, &result);
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();

        if (ok) {
            results->push_back(std::move(result));
        }

        stats_.record_frame(0, dt, 0, 0);
    }

    return !results->empty();
}

bool InferGroup::add_model(const ModelConfig& mcfg) {
    auto engine = std::make_unique<InferenceEngine>();
    if (!engine->load(mcfg)) return false;
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
            eng->unload();
            return eng->load(mcfg);
        }
    }
    return false;
}
