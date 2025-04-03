//
// Created by aichao on 2025/4/1.
//

#pragma once
#include <random>
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision {
    inline cv::Scalar get_random_color() {
        std::random_device rd; // 获取随机数种子
        std::mt19937 gen(rd()); // 使用Mersenne Twister算法生成随机数
        std::uniform_int_distribution dis(0, 255); // 定义随机数范围为1到255
        return {
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen)),
            static_cast<double>(dis(gen))
        };
    }

    MODELDEPLOY_CXX_EXPORT cv::Mat VisDetection(cv::Mat cv_image, const DetectionResult& result,
                                                const std::string& font_path, int font_size,
                                                double alpha, int save_result);

    MODELDEPLOY_CXX_EXPORT cv::Mat VisOcr(cv::Mat image, const OCRResult& results, const std::string& font_path,
                                          int font_size, double alpha, int save_result);

    MODELDEPLOY_CXX_EXPORT cv::Mat VisFaceDetection(cv::Mat cv_image, const DetectionLandmarkResult& result,
                                                    const std::string& font_path, int font_size = 14,
                                                    int landmark_radius = 4, double alpha = 0.5,
                                                    bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_lpr(cv::Mat& cv_image, const LprResult& result,
                                           const std::string& font_path, int font_size = 14,
                                           int landmark_radius = 4, double alpha = 0.5, bool save_result = false);
}
