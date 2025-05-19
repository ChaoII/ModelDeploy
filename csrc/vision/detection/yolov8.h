//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "preprocessor.h"
#include "postprocessor.h"
#include "csrc/base_model.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT YOLOv8 : public BaseModel {
    public:
        YOLOv8(const std::string& model_file,
               const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "yolov8"; }

        virtual bool predict(const cv::Mat& image, DetectionResult* result);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<DetectionResult>* results);

        virtual YOLOv8Preprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual YOLOv8Postprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        YOLOv8Preprocessor preprocessor_;
        YOLOv8Postprocessor postprocessor_;
    };
} // namespace detection
