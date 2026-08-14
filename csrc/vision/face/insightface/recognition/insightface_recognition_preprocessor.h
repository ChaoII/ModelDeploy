//
// insightface buffalo_l w600k_r50 前处理。
// norm_crop（Umeyama 相似变换 + warpAffine 到 112）+ (x-127.5)/127.5 + swapRB。
// norm_crop 含旋转，架构 fused_preprocess 不支持旋转，故 preprocessor 内用 OpenCV warpAffine。
//
#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceRecPreprocessor {
    public:
        InsightFaceRecPreprocessor();

        bool run(const ImageData& image, const std::vector<std::array<float, 2>>& kps,
                 Tensor* output) const;

        int input_size_ = 112;

    private:
        void make_blob(const cv::Mat& warped, float* dst) const;
    };

} // namespace modeldeploy::vision::face
