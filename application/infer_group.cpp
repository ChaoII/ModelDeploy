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
                             std::vector<InferResult>* results,
                             ImageData* frame_out) {
    if (!initialized_) return false;
    results->clear();

    size_t y_size = static_cast<size_t>(height) * width;
    size_t uv_size = y_size / 2;
    size_t total = y_size + uv_size;

    // 从解码器 CPU 帧数据创建连续 NV12 buffer
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

    // 对每个模型执行推理（使用 BGR 图像）
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

        InferResult result;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = engine->infer(input_image, &result);
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();

        if (ok)
            results->push_back(std::move(result));

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
