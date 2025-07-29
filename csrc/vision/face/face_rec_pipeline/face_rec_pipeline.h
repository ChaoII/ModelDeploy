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

        [[nodiscard]] bool is_initialized() const override;

    protected:
        std::unique_ptr<Scrfd> detector_ = nullptr;
        std::unique_ptr<SeetaFaceID> recognizer_ = nullptr;
    };
}
