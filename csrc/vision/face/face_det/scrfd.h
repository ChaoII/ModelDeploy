
//
// Created by aichao on 2025/3/26.
//

#pragma once
#include <unordered_map>
#include "csrc/base_model.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::face {
    /*! @brief SCRFD model object used when to load a SCRFD model exported by SCRFD.
             */
    class MODELDEPLOY_CXX_EXPORT SCRFD : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./scrfd.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit SCRFD(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "scrfd"; }
        /** \brief Predict the face detection result for an input image
         *
         * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face detection result will be writen to this structure
         * \param[in] conf_threshold confidence threashold for postprocessing, default is 0.25
         * \param[in] nms_iou_threshold iou threashold for NMS, default is 0.4
         * \return true if the prediction successed, otherwise false
         */
        bool predict(cv::Mat& im, DetectionLandmarkResult* result,
                     float conf_threshold = 0.25f,
                     float nms_iou_threshold = 0.4f);

        /// Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default (640, 640)
        std::vector<int> size{640, 640};
        /// padding value, size should be the same as channels
        std::vector<float> padding_value{0.0, 0.0, 0.0};
        /// only pad to the minimum rectange which height and width is times of stride
        bool is_mini_pad = false;
        /// while is_mini_pad = false and is_no_pad = true,
        /// will resize the image to the set size
        bool is_no_pad = false;
        /// if is_scale_up is false, the input image only can be zoom out,
        /// the maximum resize scale cannot exceed 1.0
        bool is_scale_up = true;
        /// padding stride, for is_mini_pad
        int stride = 32;

        /// Argument for image postprocessing step, downsample strides (namely, steps) for SCRFD to generate anchors, will take (8,16,32) as default values
        std::vector<int> downsample_strides{8, 16, 32};

        /// Argument for image postprocessing step, landmarks_per_face, default 5 in SCRFD
        int landmarks_per_face = 5;

        /// Argument for image postprocessing step, the outputs of onnx file with key points features or not, default true
        bool use_kps = true;

        /// Argument for image postprocessing step, the upperbound number of boxes processed by nms, default 30000
        int max_nms = 30000;

        /// Argument for image postprocessing step, anchor number of each stride, default 2
        unsigned int num_anchors = 2;


    private:
        bool Initialize();

        bool preprocess(cv::Mat* mat, Tensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info);

        bool postprocess(std::vector<Tensor>& infer_result,
                         DetectionLandmarkResult* result,
                         const std::map<std::string, std::array<float, 2>>& im_info,
                         float conf_threshold, float nms_iou_threshold);

        void generate_points();

        static void letter_box(cv::Mat* mat, const std::vector<int>& size,
                               const std::vector<float>& color, bool _auto,
                               bool scale_fill = false, bool scale_up = true,
                               int stride = 32);

        bool is_dynamic_input_{};

        bool center_points_is_update_{};

        typedef struct {
            float cx;
            float cy;
        } SCRFDPoint;

        std::unordered_map<int, std::vector<SCRFDPoint>> center_points_;
    };
}
