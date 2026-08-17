//
// Created by aichao on 2025/4/7.
//

#pragma once


#include "core/md_decl.h"
#include "base_model.h"
#include "vision/common/result.h"
#include "vision/face/face_det/scrfd.h"
#include "vision/face/face_rec/seetaface.h"

namespace modeldeploy::vision::face {
    class MODELDEPLOY_CXX_EXPORT FaceRecognizerPipeline : public BaseModel {
    public:
        FaceRecognizerPipeline(const std::string& det_model_path,
                               const std::string& rec_model_path,
                               const RuntimeOption& option = RuntimeOption());

        ~FaceRecognizerPipeline() override;

        virtual bool predict(const ImageData& image, std::vector<FaceRecognitionResult>* results,
                             TimerArray* timers = nullptr);

        // 只识别 bbox 面积最大的人脸（多人场景只取主脸）。
        // face_count 返回检测到的人脸总数（>1 表示画面多人，可作告警）。
        // 返回 false 表示画面无人脸。
        virtual bool predict_max_face(const ImageData& image, FaceRecognitionResult* result,
                                      int* face_count = nullptr,
                                      TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const override;

        /// 暴露 det 子模型（用于配置检测前/后处理参数，如 conf/nms threshold）
        std::shared_ptr<Scrfd> get_detector();

        [[nodiscard]] std::unique_ptr<FaceRecognizerPipeline> clone() const;


    protected:
        std::shared_ptr<Scrfd> detector_ = nullptr;
        std::shared_ptr<SeetaFaceID> recognizer_ = nullptr;
    };
}
