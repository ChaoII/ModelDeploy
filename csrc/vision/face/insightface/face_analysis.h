//
// insightface FaceAnalysis 综合 pipeline：det -> landmark(2d/3d) -> recognition。
// 对齐 python insightface.app.FaceAnalysis。
//
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/face/insightface/insightface_scrfd.h"
#include "vision/face/insightface/insightface_landmark.h"
#include "vision/face/insightface/insightface_recognition.h"
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
        // 传入各模型路径 + runtime option（可共享）
        InsightFaceAnalysis(const std::string& det_model,
                            const std::string& rec_model,
                            const std::string& lmk2d_model,
                            const std::string& lmk3d_model,
                            const RuntimeOption& option = RuntimeOption());

        // 单图全流程分析
        bool analyze(const ImageData& image, std::vector<InsightFaceResult>* results,
                     bool with_2d106 = true, bool with_3d68 = true,
                     bool with_recognition = true,
                     TimerArray* timers = nullptr);

        // 仅检测
        bool detect(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                    TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const;

        float det_thresh = 0.5f;

        // 便捷构造：从 buffalo_l 目录加载全部
        static std::unique_ptr<InsightFaceAnalysis> create_from_dir(
            const std::string& model_dir, const RuntimeOption& option = RuntimeOption());

    private:
        std::unique_ptr<InsightFaceDet> det_;
        std::unique_ptr<InsightFaceRecognition> rec_;
        std::unique_ptr<InsightFaceLandmark> lmk_2d_;
        std::unique_ptr<InsightFaceLandmark> lmk_3d_;
        bool initialized_ = false;
    };

} // namespace modeldeploy::vision::face
