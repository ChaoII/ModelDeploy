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
                          ImageData* out_frame = nullptr,
                          Device src_device = Device::CPU,
                          TimerArray* timers = nullptr);

        // 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）;depth 无实例框，直接返回
        bool draw_result(ImageData& frame, const DepthResult& result, double threshold = 0.5);


        [[nodiscard]] std::unique_ptr<UltralyticsDepth> clone() const;

        UltralyticsDepthPreprocessor& get_preprocessor() { return preprocessor_; }
        UltralyticsDepthPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        // NV12/device 单帧共享推理核心（预处走 plane(0)/plane(1) 零拷贝），predict/predict_nv12 共用
        bool predict_single_nv12(const ImageData& frame,
                                 DepthResult* result,
                                 LetterBoxRecord* letter_box_record,
                                 TimerArray* timers);
        UltralyticsDepthPreprocessor preprocessor_;
        UltralyticsDepthPostprocessor postprocessor_;
    };
}
