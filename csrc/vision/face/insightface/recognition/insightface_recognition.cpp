//
// insightface buffalo_l w600k_r50 ArcFace 识别模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/recognition/insightface_recognition.h"

namespace modeldeploy::vision::face {

    InsightFaceRecognition::InsightFaceRecognition(const std::string& model_file,
                                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceRecognition::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    bool InsightFaceRecognition::predict(const ImageData& image,
                                         const std::vector<std::array<float, 2>>& kps,
                                         std::vector<float>* embedding,
                                         TimerArray* timers) {
        if (!embedding) return false;
        reused_input_tensors_.resize(1);
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(image, kps, &reused_input_tensors_[0])) return false;
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, embedding)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceRecognition> InsightFaceRecognition::clone() const {
        auto clone_model = std::make_unique<InsightFaceRecognition>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
