//
// Created by aichao on 2026/1/30.
//

#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "base_model.h"
#include "vision/common/result.h"
#include "vision/common/struct.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT YoloPreprocessBase {
    public:
        YoloPreprocessBase();
        virtual ~YoloPreprocessBase() = default;

        virtual bool run(const std::vector<ImageData>& images, std::vector<Tensor>* outputs,
                         std::vector<LetterBoxRecord>* letter_box_records) const;


        [[nodiscard]] virtual std::vector<int> get_size() const { return size_; }

        virtual void set_size(const std::vector<int>& size) { size_ = size; }

        [[nodiscard]] virtual float get_padding_value() const { return padding_value_; }

        virtual void set_padding_value(float padding_value) {
            padding_value_ = padding_value;
        }

        virtual void use_cuda_preproc() { use_cuda_preproc_ = true; }

    protected:
        bool preprocess(const ImageData& image, Tensor* output, LetterBoxRecord* letter_box_record) const;

        bool use_cuda_preproc_ = false;
        std::vector<int> size_;
        float padding_value_;
    };


    class MODELDEPLOY_CXX_EXPORT YoloPostprocessorBase {
    public:
        YoloPostprocessorBase() = default;
        virtual ~YoloPostprocessorBase() = default;

        virtual bool run_without_nms(const std::vector<Tensor>& tensors,
                                     std::vector<std::vector<DetectionResult>>* results,
                                     const std::vector<LetterBoxRecord>& letter_box_records) const;

        virtual bool run_with_nms(const std::vector<Tensor>& tensors,
                                  std::vector<std::vector<DetectionResult>>* results,
                                  const std::vector<LetterBoxRecord>& letter_box_records) const;

        virtual bool run(const std::vector<Tensor>& tensors,
                         std::vector<std::vector<DetectionResult>>* results,
                         const std::vector<LetterBoxRecord>& letter_box_records) const;

        virtual void set_conf_threshold(const float& conf_threshold) {
            conf_threshold_ = conf_threshold;
        }

        [[nodiscard]] virtual float get_conf_threshold() const { return conf_threshold_; }

        virtual void set_nms_threshold(const float& nms_threshold) {
            nms_threshold_ = nms_threshold;
        }

        [[nodiscard]] virtual float get_nms_threshold() const { return nms_threshold_; }

    protected:
        float conf_threshold_ = 0.3f;
        float nms_threshold_ = 0.5f;
    };


    class MODELDEPLOY_CXX_EXPORT YoloBase : public BaseModel {
    public:
        explicit YoloBase(const std::string& model_file,
                          const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "YoloBase"; }

        virtual bool predict(const ImageData& image, std::vector<DetectionResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<DetectionResult>>* results,
                                   TimerArray* timers = nullptr);

        [[nodiscard]] virtual std::unique_ptr<YoloBase> clone() const = 0;

        virtual YoloPreprocessBase& get_preprocessor() ;

        virtual YoloPostprocessorBase& get_postprocessor() ;

    protected:
        virtual bool initialize();
        YoloPreprocessBase preprocessor_;
        YoloPostprocessorBase postprocessor_;
    };
} // namespace detection
