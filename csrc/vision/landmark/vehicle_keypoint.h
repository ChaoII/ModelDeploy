//
// Created by aichao on 2026/08/23.
//
#pragma once

#include <memory>
#include <string>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/pose/ultralytics_pose.h"

namespace modeldeploy::vision::landmark {
    /*! @brief 车辆关键点识别（默认 4 车轮关键点），薄封装 UltralyticsPose（复用 set_keypoints_num 泛化）。 */
    class MODELDEPLOY_CXX_EXPORT VehicleKeypoint {
    public:
        explicit VehicleKeypoint(const std::string& model_file,
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

        bool is_initialized() const { return pose_.is_initialized(); }

        std::unique_ptr<VehicleKeypoint> clone() const;

    private:
        vision::detection::UltralyticsPose pose_;
    };
} // namespace modeldeploy::vision::landmark
