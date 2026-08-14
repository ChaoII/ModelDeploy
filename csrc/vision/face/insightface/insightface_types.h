//
// insightface 结果类型定义。
//
#pragma once

#include <vector>
#include <array>
#include <string>
#include "core/md_decl.h"

namespace modeldeploy::vision::face {

    // 检测到的人脸：框 + 5 关键点 + 置信度
    struct MODELDEPLOY_CXX_EXPORT InsightFaceBox {
        std::array<float, 4> bbox; // x1, y1, x2, y2（原图坐标）
        std::vector<std::array<float, 2>> kps; // 5 个关键点（原图坐标）
        float score = 0.0f;
    };

    // 2D 106 关键点结果
    struct MODELDEPLOY_CXX_EXPORT InsightLandmark2D106 {
        std::vector<std::array<float, 2>> landmarks; // 106 个点（原图坐标）
    };

    // 3D 68 关键点 + 姿态
    struct MODELDEPLOY_CXX_EXPORT InsightLandmark3D68 {
        std::vector<std::array<float, 3>> landmarks; // 68 个 3D 点（原图坐标）
        std::array<float, 3> pose; // pitch, yaw, roll（角度）
    };

    // 识别结果：512 维 embedding
    struct MODELDEPLOY_CXX_EXPORT InsightRecognitionResult {
        std::vector<float> embedding;
    };

} // namespace modeldeploy::vision::face
