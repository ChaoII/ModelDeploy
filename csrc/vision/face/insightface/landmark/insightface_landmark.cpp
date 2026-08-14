//
// insightface buffalo_l landmark 模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/landmark/insightface_landmark.h"
#include "vision/face/insightface/face_align_utils.h"
#include "vision/processors/processor_factory.h"

namespace modeldeploy::vision::face {

    InsightFaceLandmark::InsightFaceLandmark(const std::string& model_file,
                                             const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceLandmark::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceLandmark::predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                                            std::vector<std::array<float, 2>>* landmarks,
                                            TimerArray* timers) {
        if (!landmarks) return false;
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
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_2d(reused_output_tensors_, inv_M, input_size_[0], landmarks)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    bool InsightFaceLandmark::predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                                           std::vector<std::array<float, 3>>* landmarks,
                                           std::array<float, 3>* pose,
                                           TimerArray* timers) {
        if (!landmarks) return false;
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
        const cv::Mat inv_M = invert_affine_transform(M);
        if (!postprocessor_.run_3d(reused_output_tensors_, inv_M, input_size_[0], landmarks, pose)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceLandmark> InsightFaceLandmark::clone() const {
        auto clone_model = std::make_unique<InsightFaceLandmark>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
