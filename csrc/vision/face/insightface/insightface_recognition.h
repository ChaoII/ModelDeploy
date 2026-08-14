//
// insightface buffalo_l w600k_r50：ArcFace 人脸识别。
// 标准架构：Preprocessor（norm_crop 5 点对齐）→ Runtime → Postprocessor。
// norm_crop 是含旋转的相似变换（Umeyama），架构无此 fused 算子，preprocessor 内用 OpenCV warpAffine。
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

    // 前处理：norm_crop（estimate_norm 相似变换 + warpAffine 到 112）+ (x-127.5)/127.5 + swapRB
    class MODELDEPLOY_CXX_EXPORT InsightFaceRecPreprocessor {
    public:
        InsightFaceRecPreprocessor();

        bool run(const ImageData& image, const std::vector<std::array<float, 2>>& kps,
                 Tensor* output) const;

        int input_size_ = 112;

    private:
        // 手写 blob（OpenCV 5 无 dnn 模块）
        void make_blob(const cv::Mat& warped, float* dst) const;
    };

    // 后处理：直接取输出向量为 embedding
    class MODELDEPLOY_CXX_EXPORT InsightFaceRecPostprocessor {
    public:
        bool run(const std::vector<Tensor>& infer_results, std::vector<float>* embedding);
    };

    class MODELDEPLOY_CXX_EXPORT InsightFaceRecognition : public BaseModel {
    public:
        explicit InsightFaceRecognition(const std::string& model_file,
                                        const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "InsightFaceRecognition"; }

        bool predict(const ImageData& image,
                     const std::vector<std::array<float, 2>>& kps,
                     std::vector<float>* embedding,
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
