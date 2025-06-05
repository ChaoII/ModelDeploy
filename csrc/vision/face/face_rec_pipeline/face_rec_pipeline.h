//
// Created by aichao on 2025/4/7.
//

#pragma once


#include "csrc/core/md_decl.h"
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/face/face_rec/seetaface.h"

namespace modeldeploy::vision::face
{
    class MODELDEPLOY_CXX_EXPORT FaceRecognizerPipeline : public BaseModel {
    public:
        FaceRecognizerPipeline(const std::string& det_model_path,
                               const std::string& rec_model_path,
                               int thread_num = 8);

        ~FaceRecognizerPipeline() override;

        virtual bool predict(const cv::Mat& image, std::vector<FaceRecognitionResult>* results,
                             TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const override;

    protected:
        std::unique_ptr<Scrfd> detector_ = nullptr;
        std::unique_ptr<SeetaFaceID> recognizer_ = nullptr;
    };
}
