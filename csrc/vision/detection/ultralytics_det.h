//
// Created by aichao on 2025/2/20.
//
#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/detection/preprocessor.h"
#include "vision/detection/postprocessor.h"

namespace modeldeploy::vision::detection {
    class MODELDEPLOY_CXX_EXPORT UltralyticsDet : public BaseModel {
    public:
        explicit UltralyticsDet(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsDet"; }

        bool predict(const ImageData& image, std::vector<DetectionResult>* result,
                     TimerArray* timers = nullptr);

        // 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）
        bool draw_result(ImageData& frame, const std::vector<DetectionResult>& result,
                         double threshold = 0.5);

        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<std::vector<DetectionResult>>* results,
                           TimerArray* timers = nullptr);

        [[nodiscard]] std::unique_ptr<UltralyticsDet> clone() const;


        virtual UltralyticsPreprocessor& get_preprocessor() {
            return preprocessor_;
        }

        virtual UltralyticsPostprocessor& get_postprocessor() {
            return postprocessor_;
        }

    protected:
        bool initialize();
        // NV12/device 单帧共享推理核心（预处走 plane(0)/plane(1) 零拷贝），predict(ImageData) 的 NV12 分支共用
        bool predict_single_nv12(const ImageData& frame,
                                 std::vector<DetectionResult>* result,
                                 LetterBoxRecord* letter_box_record,
                                 TimerArray* timers);
        UltralyticsPreprocessor preprocessor_;
        UltralyticsPostprocessor postprocessor_;
    };
} // namespace detection
