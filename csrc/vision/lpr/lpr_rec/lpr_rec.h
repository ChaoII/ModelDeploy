//
// Created by aichao on 2025/6/10.
//
#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/lpr/lpr_rec/preprocessor.h"
#include "csrc/vision/lpr/lpr_rec/postprocessor.h"

namespace modeldeploy::vision::lpr {
    class MODELDEPLOY_CXX_EXPORT LprRecognizer : public BaseModel {
    public:
        explicit LprRecognizer(const std::string& model_file,
                               const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "LprRecognizer"; }

        virtual bool predict(const cv::Mat& image, LprResult* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<cv::Mat>& images,
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
