#pragma once
#include "csrc/base_model.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision::facealign {
    /*! @brief FaceLandmark1000 model object used when to load a FaceLandmark1000 model exported by FaceLandmark1000.
     */
    class MODELDEPLOY_CXX_EXPORT FaceLandmark1000 : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./face_landmarks_1000.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        FaceLandmark1000(const std::string& model_file,
                         const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "FaceLandmark1000"; }
        /** \brief Predict the face detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face detection result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        virtual bool Predict(cv::Mat* im, FaceAlignmentResult* result);

        /** \brief Get the input size of image
         *
         * \return Vector of int values, default {128,128}
         */
        std::vector<int> GetSize() { return size_; }
        /** \brief Set the input size of image
         *
         * \param[in] size Vector of int values which represents {width, height} of image
         */
        void SetSize(const std::vector<int>& size) { size_ = size; }

    private:
        bool Initialize();

        bool Preprocess(cv::Mat* mat, MDTensor* outputs,
                        std::map<std::string, std::array<int, 2>>* im_info);

        bool Postprocess(MDTensor& infer_result, FaceAlignmentResult* result,
                         const std::map<std::string, std::array<int, 2>>& im_info);
        // tuple of (width, height), default (128, 128)
        std::vector<int> size_;
    };
} // namespace modeldeploy::vision::facealign
