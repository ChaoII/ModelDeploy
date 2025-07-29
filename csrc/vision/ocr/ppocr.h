//
// Created by aichao on 2025/2/21.
//

#pragma once

#include <vector>

#include "base_model.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/ocr/classifier.h"
#include "vision/ocr/dbdetector.h"
#include "vision/ocr/recognizer.h"

namespace modeldeploy::vision::ocr {
    class MODELDEPLOY_CXX_EXPORT PaddleOCR : public BaseModel {
    public:
        PaddleOCR(const std::string& det_model_path,
                  const std::string& cls_model_path,
                  const std::string& rec_model_path,
                  const std::string& dict_path,
                  const RuntimeOption& option = RuntimeOption());

        ~PaddleOCR() override;


        [[nodiscard]] std::string name() const override { return "PaddleOCR"; }


        virtual bool predict(const ImageData& image, OCRResult* result, TimerArray* timers = nullptr);

        /** \brief BatchPredict the input image and get OCR result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] batch_result The output list of OCR result will be writen to this structure.
         * \param timers
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool batch_predict(const std::vector<ImageData>& images,
                                   std::vector<OCRResult>* batch_result, TimerArray* timers = nullptr);

        [[nodiscard]] bool is_initialized() const override;

        bool set_cls_batch_size(int cls_batch_size);

        [[nodiscard]] int get_cls_batch_size() const;

        bool set_rec_batch_size(int rec_batch_size);

        [[nodiscard]] int get_rec_batch_size() const;

        std::shared_ptr<DBDetector> get_detector();

        std::shared_ptr<Classifier> get_classifier();

        std::shared_ptr<Recognizer> get_recognizer();

    protected:
        std::shared_ptr<DBDetector> detector_ = nullptr;
        std::shared_ptr<Classifier> classifier_ = nullptr;
        std::shared_ptr<Recognizer> recognizer_ = nullptr;

    private:
        int cls_batch_size_ = 1;
        int rec_batch_size_ = 6;
    };
}
