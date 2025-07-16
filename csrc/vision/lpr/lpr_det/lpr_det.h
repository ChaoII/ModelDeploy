//
// Created by aichao on 2025/6/10.
//
#pragma once

#include "base_model.h"
#include "vision/lpr/lpr_det/preprocessor.h"
#include "vision/lpr/lpr_det/postprocessor.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprDetection : public BaseModel {
    public:
        explicit LprDetection(const std::string& model_file,
                              const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "LprDetection"; }

        virtual bool predict(const cv::Mat& image, std::vector<DetectionLandmarkResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<DetectionLandmarkResult>>* results,
                                   TimerArray* timers = nullptr);

        virtual LprDetPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual LprDetPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        LprDetPreprocessor preprocessor_;
        LprDetPostprocessor postprocessor_;
    };
} // namespace detection
