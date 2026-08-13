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

        [[nodiscard]] std::unique_ptr<UltralyticsSem> clone() const;

        UltralyticsSemPreprocessor& get_preprocessor() { return preprocessor_; }
        UltralyticsSemPostprocessor& get_postprocessor() { return postprocessor_; }

    protected:
        bool initialize();
        UltralyticsSemPreprocessor preprocessor_;
        UltralyticsSemPostprocessor postprocessor_;
    };
}
