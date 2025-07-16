//
// Created by aichao on 2025/06/04.
//

#pragma once

#include "core/md_decl.h"
#include "vision/common/result.h"

namespace modeldeploy::vision {
    MODELDEPLOY_CXX_EXPORT void dis_cls(const ClassifyResult& result);

    MODELDEPLOY_CXX_EXPORT void dis_det(const std::vector<DetectionResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_iseg(const std::vector<InstanceSegResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_obb(const std::vector<ObbResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_ocr(const OCRResult& results);

    MODELDEPLOY_CXX_EXPORT void dis_lmk(const std::vector<DetectionLandmarkResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_lpr(const std::vector<LprResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_pose(const std::vector<PoseResult>& result);

    MODELDEPLOY_CXX_EXPORT void dis_face_rec(std::vector<FaceRecognitionResult> results);
}
