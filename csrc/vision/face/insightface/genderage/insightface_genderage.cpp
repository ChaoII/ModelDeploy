//
// insightface buffalo_l genderage 模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/genderage/insightface_genderage.h"
#include "vision/processors/processor_factory.h"

namespace modeldeploy::vision::face {

    InsightFaceGenderAge::InsightFaceGenderAge(const std::string& model_file,
                                               const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceGenderAge::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceGenderAge::predict_gender_age(const ImageData& image, const std::array<float, 4>& bbox,
                                                  GenderAgeResult* result, TimerArray* timers) {
        if (!result) return false;
        cv::Mat M;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, bbox, &M, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, result)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceGenderAge> InsightFaceGenderAge::clone() const {
        auto clone_model = std::make_unique<InsightFaceGenderAge>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
