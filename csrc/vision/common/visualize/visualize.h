//
// Created by aichao on 2025/4/1.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision {
    cv::Scalar get_random_color();

    void draw_rectangle_and_text(cv::Mat& image, cv::Rect2f box, const std::string& text,
                                 const cv::Scalar& color, cv::FontFace font, int font_size,
                                 int thickness, bool draw_text = false);

    void draw_landmarks(cv::Mat& cv_image,
                        const std::vector<cv::Point2f>& landmarks,
                        int landmark_radius);


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_cls(
        cv::Mat& cv_image,
        const ClassifyResult& result,
        int top_k = 1,
        float score_threshold = 0.5,
        const std::string& font_path = "",
        int font_size = 14,
        double alpha = 0.15, bool save_result = false);


    MODELDEPLOY_CXX_EXPORT cv::Mat vis_det(
        cv::Mat& cv_image, const std::vector<DetectionResult>& result, double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_iseg(
        cv::Mat& cv_image, const std::vector<InstanceSegResult>& result,
        double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_obb(
        cv::Mat& cv_image, const std::vector<ObbResult>& result,
        double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_ocr(
        cv::Mat& image, const OCRResult& result, const std::string& font_path,
        int font_size, double alpha, bool save_result);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_det_landmarks(
        cv::Mat cv_image, const std::vector<DetectionLandmarkResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_lpr(
        cv::Mat& cv_image, const std::vector<LprResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);

    MODELDEPLOY_CXX_EXPORT cv::Mat vis_pose(
        cv::Mat& cv_image, const std::vector<PoseResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);
}
