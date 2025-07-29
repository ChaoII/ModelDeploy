//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/detection/preprocessor.h"
#include "vision/detection/postprocessor.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsDet : public BaseModel {
    public:
        explicit UltralyticsDet(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsDet"; }

        virtual bool predict(const ImageData& image, std::vector<DetectionResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<DetectionResult>>* results,
                                   TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<UltralyticsDet> clone() const;

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
