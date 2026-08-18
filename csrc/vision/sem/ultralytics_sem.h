//
// Created by aichao on 2026/8/13.
//

#pragma once

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/sem/preprocessor.h"
#include "vision/sem/postprocessor.h"

namespace modeldeploy::vision::detection {
    /*! @brief yolo26n-sem model object (semantic segmentation, cityscapes 19 classes)
     */
    class MODELDEPLOY_CXX_EXPORT UltralyticsSem : public BaseModel {
    public:
        explicit UltralyticsSem(const std::string& model_file,
                                const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "UltralyticsSem"; }

        bool predict(const ImageData& image, SemSegResult* result, TimerArray* timers = nullptr);

        bool batch_predict(const std::vector<ImageData>& images,
                           std::vector<SemSegResult>* results, TimerArray* timers = nullptr);
        // 就地绘制结果到 frame（按 frame.device() 分发到 processor backend）;sem 无实例框，直接返回
        bool draw_result(ImageData& frame, const SemSegResult& result, double threshold = 0.5);


        [[nodiscard]] std::unique_ptr<UltralyticsSem> clone() const;

        UltralyticsSemPreprocessor& get_preprocessor() { return preprocessor_; }
        UltralyticsSemPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        // NV12/device 单帧共享推理核心（预处走 plane(0)/plane(1) 零拷贝），predict(ImageData) 的 NV12 分支共用
        bool predict_single_nv12(const ImageData& frame,
                                 SemSegResult* result,
                                 LetterBoxRecord* letter_box_record,
                                 TimerArray* timers);
        UltralyticsSemPreprocessor preprocessor_;
        UltralyticsSemPostprocessor postprocessor_;
    };
}
