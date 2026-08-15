//
// insightface buffalo_l w600k_r50：ArcFace 人脸识别模型。
// 标准架构（对齐 detection）：InsightFaceRecPreprocessor → Runtime → InsightFaceRecPostprocessor。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/recognition/insightface_recognition_preprocessor.h"
#include "vision/face/insightface/recognition/insightface_recognition_postprocessor.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceRecognition : public BaseModel {
    public:
        explicit InsightFaceRecognition(const std::string& model_file,
                                        const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceRecognition"; }

        bool predict(const ImageData& image,
                     const std::vector<std::array<float, 2>>& kps,
                     std::vector<float>* embedding,
                     TimerArray* timers = nullptr);

        // 多张脸一次 batch 推理（输入同尺寸 [N,3,112,112]，显著快于逐脸 predict）
        bool batch_predict(const ImageData& image,
                           const std::vector<std::vector<std::array<float, 2>>>& kps_list,
                           std::vector<std::vector<float>>* embeddings,
                           TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceRecognition> clone() const;

        virtual InsightFaceRecPreprocessor& get_preprocessor() { return preprocessor_; }
        virtual InsightFaceRecPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool Initialize();
        InsightFaceRecPreprocessor preprocessor_;
        InsightFaceRecPostprocessor postprocessor_;
    };

} // namespace modeldeploy::vision::face
