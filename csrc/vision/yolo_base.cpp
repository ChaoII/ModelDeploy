//
// Created by aichao on 2026/1/30.
//

//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/yolo_base.h"
#ifdef WITH_GPU
#include "vision/common/processors/yolo_preproc.cuh"
#endif
#include "vision/common/processors/yolo_preproc.h"

namespace modeldeploy::vision::detection {
    YoloPreprocessBase::YoloPreprocessBase() {
        size_ = {640, 640};
        padding_value_ = 114.0f;
    }

    bool YoloPreprocessBase::preprocess(const ImageData& image, Tensor* output,
                                        LetterBoxRecord* letter_box_record) const {
        if (use_cuda_preproc_) {
#ifdef WITH_GPU
            return yolo_preprocess_cuda(image, output, size_, padding_value_, letter_box_record);
#else
            MD_LOG_WARN << "GPU is not enabled, please compile with WITH_GPU=ON, rollback to cpu" << std::endl;
#endif
        }
        return yolo_preprocess_cpu(image, output, size_, padding_value_, letter_box_record);
    }

    bool YoloPreprocessBase::run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                                 std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) {
            MD_LOG_ERROR << "The size of input images should be greater than 0." << std::endl;
            return false;
        }
        letter_box_records->resize(images.size());
        outputs->resize(1);
        // Concat all the preprocessed data to a batch tensor
        std::vector<Tensor> tensors(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            // 修改了数据，并生成一个tensor,并记录预处理的一些参数，便于在后处理中还原
            preprocess(images[i], &tensors[i], &(*letter_box_records)[i]);
        }
        if (tensors.size() == 1) {
            (*outputs)[0] = std::move(tensors[0]);
        }
        else {
            (*outputs)[0] = std::move(Tensor::concat(tensors, 0));
        }
        return true;
    }

    YoloBase::YoloBase(const std::string& model_file, const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = YoloBase::initialize();
    }

    bool YoloBase::initialize() {
        if (!init_runtime()) {
            return false;
        }
        return true;
    }

    bool YoloBase::predict(const ImageData& image, std::vector<DetectionResult>* result,
                           TimerArray* timers) {
        std::vector<std::vector<DetectionResult>> results;
        if (!batch_predict({image}, &results, timers)) {
            return false;
        }
        *result = std::move(results[0]);
        return true;
    }


    bool YoloBase::batch_predict(const std::vector<ImageData>& images,
                                 std::vector<std::vector<DetectionResult>>* results,
                                 TimerArray* timers) {
        std::vector<LetterBoxRecord> letter_box_records;
        if (timers) timers->pre_timer.start();
        if (!preprocessor_.run(images, &reused_input_tensors_, &letter_box_records)) {
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
        if (!postprocessor_.run(reused_output_tensors_, results, letter_box_records)) {
            MD_LOG_ERROR << "Failed to postprocess the inference results by runtime." << std::endl;
            return false;
        }
        if (timers) timers->post_timer.stop();
        return true;
    }

    YoloPostprocessorBase& YoloBase::get_postprocessor() {
        return postprocessor_;
    }

    YoloPreprocessBase& YoloBase::get_preprocessor() {
        return preprocessor_;
    }


    bool YoloPostprocessorBase::run_without_nms(const std::vector<Tensor>& tensors,
                                                std::vector<std::vector<DetectionResult>>* results,
                                                const std::vector<LetterBoxRecord>& letter_box_records) const {
        MD_LOG_ERROR << "Not implement run_without_nms" << std::endl;
        return false;
    }

    bool YoloPostprocessorBase::run_with_nms(const std::vector<Tensor>& tensors,
                                             std::vector<std::vector<DetectionResult>>* results,
                                             const std::vector<LetterBoxRecord>& letter_box_records) const {
        MD_LOG_ERROR << "Not implement run_with_nms" << std::endl;
        return false;
    }

    bool YoloPostprocessorBase::run(const std::vector<Tensor>& tensors,
                                    std::vector<std::vector<DetectionResult>>* results,
                                    const std::vector<LetterBoxRecord>& letter_box_records) const {
        MD_LOG_ERROR << "Not implement run" << std::endl;
        return false;
    }
}
