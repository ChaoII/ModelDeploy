//
// Created for standalone pedestrian Re-ID (OSNet) model.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/reid/preprocessor.h"
#include "vision/reid/postprocessor.h"
#include "vision/common/result.h"

namespace modeldeploy::vision::reid {
    /*! @brief Standalone pedestrian Re-ID (OSNet) model object.
     */
    class MODELDEPLOY_CXX_EXPORT ReID : public BaseModel {
    public:
        /** \brief Set path of model file and the configuration of runtime.
         *  \param[in] model_file Path of model file, e.g ./osnet.onnx
         *  \param[in] option RuntimeOption for inference
         */
        explicit ReID(const std::string& model_file,
                      const RuntimeOption& option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "ReID"; }

        /** \brief Predict the Re-ID embedding for an input image.
         *  \param[in] img The input image data (BGR HWC, from cv::imread())
         *  \param[in] results The output embedding result
         *  \param[in] timer Optional benchmark timer
         *  \return true if the prediction succeeded, otherwise false
         */
        bool predict(const ImageData& img,
                     std::vector<ReIdResult>* results,
                     TimerArray* timer = nullptr);

        /** \brief Predict Re-ID embeddings for a batch of input images.
         *  \param[in] imgs Input image list
         *  \param[in] results Per-image embedding result list
         *  \param[in] timer Optional benchmark timer
         *  \return true if the prediction succeeded, otherwise false
         */
        bool batch_predict(const std::vector<ImageData>& imgs,
                           std::vector<std::vector<ReIdResult>>* results,
                           TimerArray* timer = nullptr);

        ReIDPreprocessor* get_preprocessor() { return &preprocessor_; }
        const ReIDPreprocessor* get_preprocessor() const { return &preprocessor_; }
        ReIDPostprocessor* get_postprocessor() { return &postprocessor_; }
        const ReIDPostprocessor* get_postprocessor() const { return &postprocessor_; }

        [[nodiscard]] std::unique_ptr<ReID> clone() const;

    private:
        bool initialize();
        void setup_processor_backend();

        ReIDPreprocessor preprocessor_;
        ReIDPostprocessor postprocessor_;
        mutable std::vector<Tensor> reused_input_tensors_;
        mutable std::vector<Tensor> reused_output_tensors_;
    };
} // namespace modeldeploy::vision::reid
