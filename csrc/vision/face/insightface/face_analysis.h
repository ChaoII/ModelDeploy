//
// insightface FaceAnalysis 综合 pipeline：det -> landmark(2d/3d) -> recognition。
// 组合多个标准子模型（每个都是 BaseModel + pre/post），多推理后端天然支持。
//
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/scrfd/insightface_scrfd.h"
#include "vision/face/insightface/landmark/insightface_landmark.h"
#include "vision/face/insightface/recognition/insightface_recognition.h"
#include "vision/face/insightface/insightface_types.h"

namespace modeldeploy::vision::face {

    // 综合人脸分析结果（对齐 insightface Face）
    struct MODELDEPLOY_CXX_EXPORT InsightFaceResult {
        std::array<float, 4> bbox; // x1,y1,x2,y2 原图坐标
        float det_score = 0.0f;
        std::vector<std::array<float, 2>> kps; // 5 关键点（原图坐标）
        std::vector<std::array<float, 2>> landmark_2d_106; // 106 个 2D 点（原图坐标）
        std::vector<std::array<float, 3>> landmark_3d_68; // 68 个 3D 点（原图坐标）
        std::array<float, 3> pose{0, 0, 0}; // pitch, yaw, roll
        std::vector<float> embedding; // 512 维
    };

    class MODELDEPLOY_CXX_EXPORT InsightFaceAnalysis {
    public:
        InsightFaceAnalysis(const std::string& det_model,
                            const std::string& rec_model,
                            const std::string& lmk2d_model,
                            const std::string& lmk3d_model,
                            const RuntimeOption& option = RuntimeOption());

        bool analyze(const ImageData& image, std::vector<InsightFaceResult>* results,
                     bool with_2d106 = true, bool with_3d68 = true,
                     bool with_recognition = true,
                     TimerArray* timers = nullptr);

        bool detect(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                    TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const;

        // 检测阈值（透传给 det preprocessor）
        void set_det_thresh(float thresh);

        static std::unique_ptr<InsightFaceAnalysis> create_from_dir(
            const std::string& model_dir, const RuntimeOption& option = RuntimeOption());

        // 访问子模型（供高级用法）
        [[nodiscard]] InsightFaceDet* det() { return det_.get(); }
        [[nodiscard]] InsightFaceRecognition* rec() { return rec_.get(); }
        [[nodiscard]] InsightFaceLandmark* lmk_2d() { return lmk_2d_.get(); }
        [[nodiscard]] InsightFaceLandmark* lmk_3d() { return lmk_3d_.get(); }

    private:
        std::unique_ptr<InsightFaceDet> det_;
        std::unique_ptr<InsightFaceRecognition> rec_;
        std::unique_ptr<InsightFaceLandmark> lmk_2d_;
        std::unique_ptr<InsightFaceLandmark> lmk_3d_;
        bool initialized_ = false;
    };

} // namespace modeldeploy::vision::face
