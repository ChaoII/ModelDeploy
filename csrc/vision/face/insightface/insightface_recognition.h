//
// insightface buffalo_l w600k_r50：ArcFace 人脸识别。
// 与 python insightface.model_zoo.arcface_onnx.ArcFaceONNX 逐值对齐。
//
#pragma once

#include <string>
#include <vector>
#include <array>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/insightface_types.h"
#include "vision/face/insightface/face_align_utils.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceRecognition : public BaseModel {
    public:
        explicit InsightFaceRecognition(const std::string& model_file,
                                        const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceRecognition"; }

        // 计算 embedding：输入 BGR 原图 + 5 个关键点（人脸对齐后裁剪到 112）
        bool predict(const ImageData& image,
                     const std::vector<std::array<float, 2>>& kps,
                     std::vector<float>* embedding,
                     TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<InsightFaceRecognition> clone() const;

        int input_size_ = 112;

    protected:
        bool Initialize();
    };

} // namespace modeldeploy::vision::face
