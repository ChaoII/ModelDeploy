//
// Created by aichao on 2025/2/24.
//


#include "core/md_log.h"
#include "vision/classification/classification.h"
#include "vision/processors/processor_factory.h"


namespace modeldeploy::vision::classification {
    Classification::Classification(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool Classification::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        // 与其它 yolo 模型一致：按运行时选择预处理后端（GPU→CUDA，CPU→CPU）。
        // 否则固定用 CPU 预处理，GPU/TRT 时输入是 CPU 张量，需额外 host->device 上传。
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool Classification::predict(const ImageData& im, ClassifyResult* result) {
        std::vector<ClassifyResult> results;
        if (!batch_predict({im}, &results)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }

    bool Classification::batch_predict(const std::vector<ImageData>& images, std::vector<ClassifyResult>* results) {
        std::vector<ImageData> _images = images;
        if (!preprocessor_.run(&_images, &reused_input_tensors_)) {
            MD_LOG_ERROR << "Failed to preprocess the input image." << std::endl;
            return false;
        }

        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference by runtime." << std::endl;
            return false;
        }

        if (!postprocessor_.run(reused_output_tensors_, results)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        return true;
    }

    std::unique_ptr<Classification> Classification::clone() const {
        auto clone_model = std::make_unique<Classification>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }
} // namespace classification

