//
// Created by aichao on 2025/4/1.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision
{
    cv::Scalar get_random_color();

    void draw_rectangle_and_text(cv::Mat& image, cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, cv::FontFace font, int font_size,
                                 int thickness, bool draw_text = false);

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<cv::Point2f>& landmarks,
                        int landmark_radius);


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_classification(
        cv::Mat& cv_image,
        const ClassifyResult& result,
        int top_k,
        float score_threshold,
        const std::string& font_path,
        int font_size,
        double alpha, bool save_result);


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_detection(
        cv::Mat& cv_image, const DetectionResult& result, double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.5, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_ocr(
        cv::Mat& image, const OCRResult& results, const std::string& font_path,
        int font_size, double alpha, bool save_result);

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

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_pose(
        cv::Mat& cv_image, const PoseResult& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.5,
        bool save_result = false);
}
