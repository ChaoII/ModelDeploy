//
// Created by aichao on 2025/2/21.
//

#pragma once

#include "core/md_decl.h"
#include "base_model.h"
#include "vision/common/result.h"
#include "vision/lpr/lpr_det/lpr_det.h"
#include "vision/lpr/lpr_rec/lpr_rec.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprPipeline : public BaseModel {
    public:
        LprPipeline(const std::string& det_model_path,
                    const std::string& rec_model_path,
                    RuntimeOption option = RuntimeOption());

        ~LprPipeline() override;

        virtual bool predict(const ImageData& image, std::vector<LprResult>* results, TimerArray* times = nullptr);

        [[nodiscard]] bool is_initialized() const override;

    protected:
        std::unique_ptr<LprDetection> detector_ = nullptr;
        std::unique_ptr<LprRecognizer> recognizer_ = nullptr;
    };
}
