//
// Created by aichao on 2025/06/2.
//
#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/pose/preprocessor.h"
#include "csrc/vision/pose/postprocessor.h"

namespace modeldeploy::vision::detection
{
    class MODELDEPLOY_CXX_EXPORT UltralyticsPose : public BaseModel {
    public:
        explicit UltralyticsPose(const std::string& model_file,
                                 const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsPose"; }

        virtual bool predict(const cv::Mat& image, PoseResult* result);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<PoseResult>* results);

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
