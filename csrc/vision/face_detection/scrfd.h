#pragma once
#include <unordered_map>
#include "csrc/base_model.h"
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::facedet {
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

        [[nodiscard]] std::string name() const { return "scrfd"; }
        /** \brief Predict the face detection result for an input image
                 *
                 * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
                 * \param[in] result The output face detection result will be writen to this structure
                 * \param[in] conf_threshold confidence threashold for postprocessing, default is 0.25
                 * \param[in] nms_iou_threshold iou threashold for NMS, default is 0.4
                 * \return true if the prediction successed, otherwise false
                 */
        virtual bool Predict(cv::Mat* im, FaceDetectionResult* result,
                             float conf_threshold = 0.25f,
                             float nms_iou_threshold = 0.4f);

        /*! @brief
                Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default (640, 640)
                */
        std::vector<int> size;
        // padding value, size should be the same as channels

        std::vector<float> padding_value;
        // only pad to the minimum rectange which height and width is times of stride
        bool is_mini_pad;
        // while is_mini_pad = false and is_no_pad = true,
        // will resize the image to the set size
        bool is_no_pad;
        // if is_scale_up is false, the input image only can be zoom out,
        // the maximum resize scale cannot exceed 1.0
        bool is_scale_up;
        // padding stride, for is_mini_pad
        int stride;
        /*! @brief
                Argument for image postprocessing step, downsample strides (namely, steps) for SCRFD to generate anchors, will take (8,16,32) as default values
                */
        std::vector<int> downsample_strides;
        /*! @brief
                Argument for image postprocessing step, landmarks_per_face, default 5 in SCRFD
                */
        int landmarks_per_face;
        /*! @brief
                Argument for image postprocessing step, the outputs of onnx file with key points features or not, default true
                */
        bool use_kps;
        /*! @brief
                Argument for image postprocessing step, the upperbond number of boxes processed by nms, default 30000
                */
        int max_nms;
        /*! @brief
                Argument for image postprocessing step, anchor number of each stride, default 2
                */
        unsigned int num_anchors;

        /// This function will disable normalize and hwc2chw in preprocessing step.
        void DisableNormalize();

        /// This function will disable hwc2chw in preprocessing step.
        void DisablePermute();

    private:
        bool Initialize();

        bool Preprocess(cv::Mat* mat, MDTensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info);

        bool Postprocess(std::vector<MDTensor>& infer_result,
                         FaceDetectionResult* result,
                         const std::map<std::string, std::array<float, 2>>& im_info,
                         float conf_threshold, float nms_iou_threshold);

        void GeneratePoints();

        void LetterBox(cv::Mat* mat, const std::vector<int>& size,
                       const std::vector<float>& color, bool _auto,
                       bool scale_fill = false, bool scale_up = true,
                       int stride = 32);

        bool is_dynamic_input_;

        bool center_points_is_update_;

        typedef struct {
            float cx;
            float cy;
        } SCRFDPoint;

        std::unordered_map<int, std::vector<SCRFDPoint>> center_points_;

        // for recording the switch of normalize
        bool disable_normalize_ = false;
        // for recording the switch of hwc2chw
        bool disable_permute_ = false;
    };
}
