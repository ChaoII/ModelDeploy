//
// Created by aichao on 2025/3/21.
//

#pragma once

#include <vector>
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/ocr/structurev2_table.h"
#include "csrc/vision/ocr/dbdetector.h"
#include "csrc/vision/ocr/recognizer.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"


/** \brief This pipeline can launch detection model, classification model and recognition model sequentially. All OCR pipeline APIs are defined inside this namespace.
 *
 */
namespace modeldeploy::vision::ocr {
    /*! @brief PPStructureV2Table is used to load PP-OCRv2 series models provided by PaddleOCR.
     */
    class MODELDEPLOY_CXX_EXPORT PPStructureV2Table : public BaseModel {
    public:
        /** \brief Set up the detection model path, recognition model path and table model path respectively.
         *
         * \param det_model_file The detection model path.
         * \param rec_model_file The recognition model path.
         * \param table_model_file The table model path.
         * \param rec_label_file The recognition label path.
         * \param table_char_dict_path The table model path.
         * \param thread_num The number of threads to use.
         * \param max_side_len The maximum side length of input image.
         * \param det_db_thresh
         * \param det_db_box_thresh
         * \param det_db_unclip_ratio
         * \param det_db_score_mode
         * \param use_dilation
         * \param rec_batch_size
         */
        PPStructureV2Table(const std::string& det_model_file,
                           const std::string& rec_model_file,
                           const std::string& table_model_file,
                           const std::string& rec_label_file,
                           const std::string& table_char_dict_path,
                           int thread_num = 8,
                           int max_side_len = 960,
                           double det_db_thresh = 0.3,
                           double det_db_box_thresh = 0.6,
                           double det_db_unclip_ratio = 1.5,
                           const std::string& det_db_score_mode = "slow",
                           bool use_dilation = false,
                           int rec_batch_size = 8);


        /** \brief Predict the input image and get OCR result.
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] result The output OCR result will be writen to this structure.
         * \return true if the prediction successed, otherwise false.
         */
        virtual bool predict(cv::Mat* image, modeldeploy::vision::OCRResult* result);

        virtual bool predict(const cv::Mat& image, modeldeploy::vision::OCRResult* result);

        /** \brief BatchPredict the input image and get OCR result.
         *
         * \param[in] images The list of input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format.
         * \param[in] batch_result The output list of OCR result will be writen to this structure.
         * \return true if the prediction successfully otherwise false.
         */
        virtual bool batch_predict(const std::vector<cv::Mat>& images,
                                   std::vector<modeldeploy::vision::OCRResult>* batch_result);

        [[nodiscard]] bool is_initialized() const override;

        bool set_rec_batch_size(int rec_batch_size);

        [[nodiscard]] int get_rec_batch_size() const;

    protected:
        std::unique_ptr<DBDetector> detector_ = nullptr;
        std::unique_ptr<Recognizer> recognizer_ = nullptr;
        std::unique_ptr<StructureV2Table> table_ = nullptr;

    private:
        int rec_batch_size_ = 6;
    };
} // namespace modeldeploy::pipeline
