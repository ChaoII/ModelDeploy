//
// insightface buffalo_l genderage 模型。
// 标准架构：InsightFaceGenderAgePreprocessor（backend）→ Runtime → InsightFaceGenderAgePostprocessor。
//
#pragma once

#include <string>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/genderage/insightface_genderage_preprocessor.h"
#include "vision/face/insightface/genderage/insightface_genderage_postprocessor.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceGenderAge : public BaseModel {
    public:
        explicit InsightFaceGenderAge(const std::string& model_file,
                                      const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceGenderAge"; }

        bool predict_gender_age(const ImageData& image, const std::array<float, 4>& bbox,
                                GenderAgeResult* result, TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceGenderAge> clone() const;

        virtual InsightFaceGenderAgePreprocessor& get_preprocessor() { return preprocessor_; }
        virtual InsightFaceGenderAgePostprocessor& get_postprocessor() { return postprocessor_; }

        std::vector<int> input_size_{96, 96};

    protected:
        bool Initialize();
        InsightFaceGenderAgePreprocessor preprocessor_;
        InsightFaceGenderAgePostprocessor postprocessor_;
    };

} // namespace modeldeploy::vision::face
