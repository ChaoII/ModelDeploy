//
// Created by aichao on 2026/08/22.
//
#pragma once

#include <memory>
#include <string>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/pose/ultralytics_pose.h"

namespace modeldeploy::vision::hand {
    /*! @brief 手部关键点识别（21 点，MediaPipe 风格），薄封装 UltralyticsPose */
    class MODELDEPLOY_CXX_EXPORT HandKeypoint {
    public:
        explicit HandKeypoint(const std::string& model_file,
                              const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr);

        bool batch_predict(const std::vector<ImageData>& imgs,
                           std::vector<std::vector<KeyPointsResult>>* results,
                           TimerArray* timer = nullptr);

        bool draw_result(ImageData& img,
                         const std::vector<KeyPointsResult>& results,
                         double threshold = 0.5);

        vision::detection::UltralyticsPosePreprocessor& get_preprocessor() {
            return pose_.get_preprocessor();
        }

        vision::detection::UltralyticsPosePostprocessor& get_postprocessor() {
            return pose_.get_postprocessor();
        }

        std::unique_ptr<HandKeypoint> clone() const;

    private:
        vision::detection::UltralyticsPose pose_;
    };
} // namespace modeldeploy::vision::hand
