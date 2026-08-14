//
// insightface buffalo_l det_10g：SCRFD 人脸检测模型实现。
//
#include "core/md_log.h"
#include "vision/face/insightface/scrfd/insightface_scrfd.h"
#include "vision/processors/processor_factory.h"

namespace modeldeploy::vision::face {

    InsightFaceDet::InsightFaceDet(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool InsightFaceDet::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceDet::predict(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                                 TimerArray* timers) {
        std::vector<std::vector<InsightFaceBox>> results;
        if (!batch_predict({image}, &results, timers)) return false;
        *boxes = std::move(results[0]);
        return true;
    }

    bool InsightFaceDet::batch_predict(const std::vector<ImageData>& images,
                                       std::vector<std::vector<InsightFaceBox>>* boxes,
                                       TimerArray* timers) {
        std::vector<ImageData> _images = images;
        std::vector<LetterBoxRecord> lbrs;
        if (timers) timers->pre_timer.start();
        reused_input_tensors_.resize(1);
        if (!preprocessor_.run(_images, &reused_input_tensors_[0], &lbrs)) {
            MD_LOG_ERROR << "Failed to preprocess." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, lbrs, boxes)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceDet> InsightFaceDet::clone() const {
        auto clone_model = std::make_unique<InsightFaceDet>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
