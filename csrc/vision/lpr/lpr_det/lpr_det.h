#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision::lpr {
    /*! @brief YOLOv5Face model object used when to load a CarPlateDetection model based on  YOLOv5Face.
     */
    class MODELDEPLOY_CXX_EXPORT LprDetection : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./yolov5face.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit LprDetection(const std::string& model_file,
                              const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "car-plate-detection"; }

        /** \brief Predict the face detection result for an input image
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face detection result will be writen to this structure
         * \param[in] conf_threshold confidence threashold for postprocessing, default is 0.25
         * \param[in] nms_iou_threshold iou threashold for NMS, default is 0.5
         * \return true if the prediction successed, otherwise false
         */
        bool predict(const cv::Mat& image, std::vector<DetectionLandmarkResult>* result,
                     float conf_threshold = 0.25, float nms_iou_threshold = 0.5);

        /*! @brief
        Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default size = {640, 640}
        */
        std::vector<int> size{640, 640};
        /// padding value, size should be the same as channels
        std::vector<float> padding_value;
        /// only pad to the minimum rectangle which height and width is times of stride
        bool is_mini_pad{};
        /// while is_mini_pad = false and is_no_pad = true,
        /// will resize the image to the set size
        bool is_no_pad{};
        /// if is_scale_up is false, the input image only can be zoom out,
        /// the maximum resize scale cannot exceed 1.0
        bool is_scale_up{};
        /// padding stride, for is_mini_pad
        int stride{};
        /*! @brief
          Argument for image postprocessing step, setup the number of landmarks for per car plate (if have), default 4.
          The output tensor's shape must be:
          (1,n,4+1+2*landmarks_per_card+1=box+obj+landmarks+cls), default 5
        */
        int landmarks_per_card{};

    private:
        bool initialize();

        bool preprocess(cv::Mat& mat, Tensor* outputs,
                        std::map<std::string, std::array<float, 2>>* im_info);

        bool postprocess(const Tensor& infer_result, std::vector<DetectionLandmarkResult>* result,
                         const std::map<std::string, std::array<float, 2>>& im_info,
                         float conf_threshold, float nms_iou_threshold) const;

        [[nodiscard]] bool is_dynamic_input() const { return is_dynamic_input_; }

        bool is_dynamic_input_{};
    };
} // namespace modeldeploy::vision::facedet
