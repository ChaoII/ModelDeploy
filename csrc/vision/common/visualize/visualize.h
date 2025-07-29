//
// Created by aichao on 2025/4/1.
//

#pragma once

#include "core/md_decl.h"
#include "vision/common/result.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision {

    MODELDEPLOY_CXX_EXPORT ImageData vis_cls(
        ImageData& image,
        const ClassifyResult& result,
        int top_k = 1,
        float score_threshold = 0.5,
        const std::string& font_path = "",
        int font_size = 14,
        double alpha = 0.15, bool save_result = false);


    MODELDEPLOY_CXX_EXPORT ImageData vis_det(
        ImageData& image, const std::vector<DetectionResult>& result, double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_iseg(
        ImageData& image, const std::vector<InstanceSegResult>& result,
        double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_obb(
        ImageData& image, const std::vector<ObbResult>& result,
        double threshold = 0.5,
        const std::string& font_path = "", int font_size = 14,
        double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_ocr(
        ImageData& image, const OCRResult& result, const std::string& font_path,
        int font_size = 14, double alpha = 0.15, bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_det_landmarks(
        ImageData& image, const std::vector<DetectionLandmarkResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_lpr(
        ImageData& image, const std::vector<LprResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);

    MODELDEPLOY_CXX_EXPORT ImageData vis_pose(
        ImageData& image, const std::vector<PoseResult>& result,
        const std::string& font_path, int font_size = 14,
        int landmark_radius = 4, double alpha = 0.15,
        bool save_result = false);
}
