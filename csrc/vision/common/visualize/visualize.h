//
// Created by aichao on 2025/4/1.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision {
    cv::Scalar get_random_color();


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_classification(
        cv::Mat& cv_image,
        const ClassifyResult& result,
        int top_k,
        float score_threshold,
        const std::string& font_path,
        int font_size,
        double alpha, int save_result);


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_detection(
        cv::Mat& cv_image, const DetectionResult& result,
        const std::string& font_path, int font_size,
        double alpha, int save_result);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_ocr(
        cv::Mat& image, const OCRResult& results, const std::string& font_path,
        int font_size, double alpha, int save_result);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_det_landmarks(
        cv::Mat cv_image, const DetectionLandmarkResult& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.5,
        bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_lpr(
        cv::Mat& cv_image, const LprResult& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.5,
        bool save_result = false);
}
