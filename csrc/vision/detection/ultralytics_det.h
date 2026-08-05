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

        /// GPU 直通：NV12 输入（host 或 device 指针）→ GPU letterbox/normalize → 推理 → 后处理。
        /// 跳过 nv12→BGR→host 的中间转换，前处理全程在 GPU（配合 use_cuda_preproc）。
        bool predict_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                          int width, int height, int step_y, int step_uv,
                          std::vector<DetectionResult>* result,
                          LetterBoxRecord* letter_box_record,
                          TimerArray* timers = nullptr);

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
        UltralyticsPreprocessor preprocessor_;
        UltralyticsPostprocessor postprocessor_;
        // Sophgo 零拷贝路径持有的输出 bm_image（attach 到输入设备内存），用 md_bmcv_image_destroy 释放
        void* out_img_ = nullptr;
    };
} // namespace detection
