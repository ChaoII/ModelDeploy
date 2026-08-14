//
// insightface buffalo_l landmark 后处理。
// landmark 输出 (+1)*scale -> 逆仿射回原图；3D 额外姿态估计。
//
#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include "core/tensor.h"
#include "core/md_decl.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceLandmarkPostprocessor {
    public:
        // 2D 106 点
        bool run_2d(const std::vector<Tensor>& infer_results, const cv::Mat& inv_M,
                    const int input_size, std::vector<std::array<float, 2>>* landmarks);
        // 3D 68 点 + 姿态
        bool run_3d(const std::vector<Tensor>& infer_results, const cv::Mat& inv_M,
                    const int input_size, std::vector<std::array<float, 3>>* landmarks,
                    std::array<float, 3>* pose);
    };

} // namespace modeldeploy::vision::face
