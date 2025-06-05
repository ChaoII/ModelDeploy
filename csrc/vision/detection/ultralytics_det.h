//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/detection/preprocessor.h"
#include "csrc/vision/detection/postprocessor.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsDet : public BaseModel {
    public:
        explicit UltralyticsDet(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsDet"; }

        virtual bool predict(const cv::Mat& image, std::vector<DetectionResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<DetectionResult>>* results,
                                   TimerArray* timers = nullptr);

        virtual UltralyticsPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual UltralyticsPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        UltralyticsPreprocessor preprocessor_;
        UltralyticsPostprocessor postprocessor_;
    };
} // namespace detection
