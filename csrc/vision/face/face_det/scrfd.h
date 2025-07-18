//
// Created by aichao on 2025/2/20.
//
#pragma once


#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/face_det/preprocessor.h"
#include "vision/face/face_det/postprocessor.h"

namespace modeldeploy::vision::face {
    class MODELDEPLOY_CXX_EXPORT Scrfd : public BaseModel {
    public:
        explicit Scrfd(const std::string& model_file,
                       const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "Scrfd"; }

        virtual bool predict(const ImageData& image, std::vector<DetectionLandmarkResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<DetectionLandmarkResult>>* results,
                                   TimerArray* timers = nullptr);

        virtual ScrfdPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual ScrfdPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        ScrfdPreprocessor preprocessor_;
        ScrfdPostprocessor postprocessor_;
    };
} // namespace detection
