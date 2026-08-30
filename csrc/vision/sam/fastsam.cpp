#include <algorithm>
#include <set>

#include "core/md_log.h"
#include "vision/sam/fastsam.h"
#include "vision/utils.h"

namespace modeldeploy::vision::seg {
    FastSam::FastSam(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool FastSam::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy backend." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool FastSam::predict(const ImageData& image, std::vector<InstanceSegResult>* result,
                          TimerArray* timers) {
        std::vector<std::vector<InstanceSegResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool FastSam::predict_with_prompts(const ImageData& image, const FastSamPrompts& prompts,
                                       std::vector<InstanceSegResult>* result, TimerArray* timers) {
        std::vector<InstanceSegResult> all;
        if (!predict(image, &all, timers)) return false;
        if (prompts.empty()) { *result = std::move(all); return true; }

        auto mask_hit = [](const InstanceSegResult& r, float px, float py) -> bool {
            if (r.mask.shape.size() < 2) return false;
            const int h = (int)r.mask.shape[0], w = (int)r.mask.shape[1];
            if (h <= 0 || w <= 0) return false;
            const int mx = (int)(px - r.box.x), my = (int)(py - r.box.y);
            if (mx < 0 || my < 0 || mx >= w || my >= h) return false;
            return r.mask.buffer[my * w + mx] != 0;
        };

        std::set<int> keep;
        for (const auto& q : prompts.bboxes) {
            int best = -1; float biou = 0.0f;
            for (int i = 0; i < (int)all.size(); ++i) {
                const float v = utils::iou_rects(q, all[i].box);
                if (v > biou) { biou = v; best = i; }
            }
            if (best >= 0 && biou > 0.0f) keep.insert(best);
        }
        for (size_t k = 0; k < prompts.points.size(); ++k) {
            const bool fg = (k < prompts.point_labels.size()) ? (prompts.point_labels[k] != 0) : true;
            const Point2f& p = prompts.points[k];
            for (int i = 0; i < (int)all.size(); ++i) {
                if (!mask_hit(all[i], p.x, p.y)) continue;
                if (fg) keep.insert(i); else keep.erase(i);
            }
        }
        result->clear(); result->reserve(keep.size());
        for (int idx : keep) result->push_back(all[idx]);
        return true;
    }

    bool FastSam::batch_predict(const std::vector<ImageData>& images,
                                std::vector<std::vector<InstanceSegResult>>* results,
                                TimerArray* timers) {
        std::vector<LetterBoxRecord> ims_info;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, &ims_info)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, results, ims_info)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<FastSam> FastSam::clone() const {
        auto clone_model = std::make_unique<FastSam>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

    bool FastSam::draw_result(ImageData& frame, const std::vector<InstanceSegResult>& result,
                              double threshold) {
        if (frame.empty()) return false;
        auto* backend = preprocessor_.get_processor_backend().get();
        if (!backend) return false;
        for (const auto& r : result) {
            if (r.score < threshold) continue;
            const Rect2f& box = r.box;
            if (!backend->draw_rect_nv12(frame, box.x, box.y, box.width, box.height,
                                         0, 165, 255, 2)) return false;
            const std::string label = std::to_string(r.label_id) + " " + std::to_string(r.score);
            backend->draw_text_nv12(frame, box.x, box.y - 16, label, 255, 255, 255, 1);
        }
        return true;
    }
} // namespace modeldeploy::vision::seg
