//
// Created by aichao on 2025/6/10.
//
#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/lpr/lpr_rec/preprocessor.h"
#include "vision/lpr/lpr_rec/postprocessor.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprRecognizer : public BaseModel {
    public:
        explicit LprRecognizer(const std::string& model_file,
                               const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "LprRecognizer"; }

        virtual bool predict(const ImageData& image, LprResult* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<LprResult>* results,
                                   TimerArray* timers = nullptr);

        virtual LprRecPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual LprRecPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        LprRecPreprocessor preprocessor_;
        LprRecPostprocessor postprocessor_;
    };
} // namespace detection
