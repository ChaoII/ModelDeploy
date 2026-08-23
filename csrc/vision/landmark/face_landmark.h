//
// Created by aichao on 2026/08/23.
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/face/insightface/landmark/insightface_landmark.h"

namespace modeldeploy::vision::landmark {
    /*! @brief 面部 Landmark 独立访问（InsightFace 2d106，106 点）。
     *  输入：人脸裁剪图；输出：KeyPointsResult（106 个 Point3f，z=0）。
     *  insightface 106 模型为仿射对齐预/后处理，与 pose letterbox 后端不兼容，
     *  故走 spec §12 专用实现出口：内部复用已建的 face::InsightFaceLandmark，仅做薄适配。
     */
    class MODELDEPLOY_CXX_EXPORT FaceLandmark {
    public:
        explicit FaceLandmark(const std::string& model_file,
                              const RuntimeOption& option = RuntimeOption());

        bool predict(const ImageData& img,
                     std::vector<KeyPointsResult>* results,
                     TimerArray* timer = nullptr);

        bool is_initialized() const;

        std::unique_ptr<FaceLandmark> clone() const;

    private:
        explicit FaceLandmark(std::unique_ptr<face::InsightFaceLandmark> lm);
        std::unique_ptr<face::InsightFaceLandmark> landmark_;
    };
} // namespace modeldeploy::vision::landmark
