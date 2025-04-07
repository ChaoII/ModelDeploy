//
// Created by aichao on 2025/2/21.
//

#pragma once

#include "csrc/core/md_decl.h"
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/lpr/lpr_det/lpr_det.h"
#include "csrc/vision/lpr/lpr_rec/lpr_rec.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprPipeline : public BaseModel {
    public:
        LprPipeline(const std::string& det_model_path,
                    const std::string& rec_model_path,
                    int thread_num = 8);

        ~LprPipeline() override;

        virtual bool predict(const cv::Mat& image, LprResult* results);

        [[nodiscard]] bool is_initialized() const override;

    protected:
        std::unique_ptr<LprDetection> detector_ = nullptr;
        std::unique_ptr<LprRecognizer> recognizer_ = nullptr;
    };
}
