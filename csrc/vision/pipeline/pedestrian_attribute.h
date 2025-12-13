//
// Created by aichao on 2025/12/13.
//

#pragma once

#include <vector>

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/classification/classification.h"
#include "vision/detection/ultralytics_det.h"

namespace modeldeploy::vision::pipeline {
    class MODELDEPLOY_CXX_EXPORT PedestrianAttribute : public BaseModel {
    public:
        PedestrianAttribute(const std::string& det_model_path,
                            const std::string& mlc_model_path,
                            const RuntimeOption& option = RuntimeOption());

        ~PedestrianAttribute() override;


        [[nodiscard]] std::string name() const override { return "PedestrianAttribute"; }


        virtual bool predict(const ImageData& image, std::vector<AttributeResult>* result,
                             TimerArray* timers = nullptr);

        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<std::vector<AttributeResult>>* batch_result,
                                   TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const override;

        void set_det_threshold(float threshold) const;

        [[nodiscard]] float get_det_threshold() const;

        void set_det_input_size(const std::vector<int>& size) const;

        [[nodiscard]] std::vector<int> get_det_input_size() const;

        void set_cls_input_size(const std::vector<int>& size) const;

        [[nodiscard]] std::vector<int> get_cls_input_size() const;

        bool set_cls_batch_size(int cls_batch_size);

        [[nodiscard]] int get_cls_batch_size() const;

        std::shared_ptr<detection::UltralyticsDet> get_detector();

        std::shared_ptr<classification::Classification> get_classifier();

    protected:
        std::shared_ptr<detection::UltralyticsDet> detector_ = nullptr;
        std::shared_ptr<classification::Classification> classifier_ = nullptr;

    private:
        int cls_batch_size_ = 8;
    };
}
