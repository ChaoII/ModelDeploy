//
// Created by aichao on 2025/06/2.
//
#pragma once


#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/pose/preprocessor.h"
#include "vision/pose/postprocessor.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsPose : public BaseModel {
    public:
        explicit UltralyticsPose(const std::string& model_file,
                                 const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsPose"; }

        virtual bool predict(const ImageData& image, std::vector<PoseResult>* result, TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<PoseResult>>* results, TimerArray* timers = nullptr);

        virtual UltralyticsPosePreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual UltralyticsPosePostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        UltralyticsPosePreprocessor preprocessor_;
        UltralyticsPosePostprocessor postprocessor_;
    };
} // namespace detection
