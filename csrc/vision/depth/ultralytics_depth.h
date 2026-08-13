//
// Created by aichao on 2026/8/13.
//

#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/depth/preprocessor.h"
#include "vision/depth/postprocessor.h"

namespace modeldeploy::vision::detection {
    /*! @brief yolo26n-depth model object (depth estimation, depth-log)
     */
    class MODELDEPLOY_CXX_EXPORT UltralyticsDepth : public BaseModel {
    public:
        explicit UltralyticsDepth(const std::string& model_file,
                                  const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsDepth"; }

        bool predict(const ImageData& image, DepthResult* result, TimerArray* timers = nullptr);

        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<DepthResult>* results, TimerArray* timers = nullptr);
        bool predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                          int width, int height, int step_y, int step_uv,
                          DepthResult* result, LetterBoxRecord* letter_box_record = nullptr,
                          TimerArray* timers = nullptr);


        [[nodiscard]] std::unique_ptr<UltralyticsDepth> clone() const;

        UltralyticsDepthPreprocessor& get_preprocessor() { return preprocessor_; }
        UltralyticsDepthPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        UltralyticsDepthPreprocessor preprocessor_;
        UltralyticsDepthPostprocessor postprocessor_;
    };
}
