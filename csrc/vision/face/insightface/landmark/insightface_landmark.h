//
// insightface buffalo_l landmark 模型：2d106det + 1k3d68。
// 标准架构（对齐 detection）：InsightFaceLandmarkPreprocessor（backend）→ Runtime → InsightFaceLandmarkPostprocessor。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/landmark/insightface_landmark_preprocessor.h"
#include "vision/face/insightface/landmark/insightface_landmark_postprocessor.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmark : public BaseModel {
    public:
        explicit InsightFaceLandmark(const std::string& model_file,
                                     const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceLandmark"; }

        bool predict_2d106(const ImageData& image, const std::array<float, 4>& bbox,
                           std::vector<std::array<float, 2>>* landmarks,
                           TimerArray* timers = nullptr);
        bool predict_3d68(const ImageData& image, const std::array<float, 4>& bbox,
                          std::vector<std::array<float, 3>>* landmarks,
                          std::array<float, 3>* pose,
                          TimerArray* timers = nullptr);

        // 多张脸一次 batch 推理（输入 [N,3,192,192]）
        bool batch_predict_2d106(const ImageData& image,
                                 const std::vector<std::array<float, 4>>& bboxes,
                                 std::vector<std::vector<std::array<float, 2>>>* landmarks_list,
                                 TimerArray* timers = nullptr);
        bool batch_predict_3d68(const ImageData& image,
                                const std::vector<std::array<float, 4>>& bboxes,
                                std::vector<std::vector<std::array<float, 3>>>* landmarks_list,
                                std::vector<std::array<float, 3>>* poses,
                                TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceLandmark> clone() const;

        virtual InsightFaceLandmarkPreprocessor& get_preprocessor() { return preprocessor_; }
        virtual InsightFaceLandmarkPostprocessor& get_postprocessor() { return postprocessor_; }

        std::vector<int> input_size_{192, 192};

    protected:
        bool Initialize();
        InsightFaceLandmarkPreprocessor preprocessor_;
        InsightFaceLandmarkPostprocessor postprocessor_;
    };

} // namespace modeldeploy::vision::face
