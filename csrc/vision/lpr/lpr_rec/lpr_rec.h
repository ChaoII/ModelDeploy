#pragma once

#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"


namespace modeldeploy::vision::lpr {
    /*! @brief YOLOv5Face model object used when to load a YOLOv5Face model exported by YOLOv5Face.
     */
    class MODELDEPLOY_CXX_EXPORT LprRecognizer : public BaseModel {
    public:
        /** \brief  Set path of model file and the configuration of runtime.
         *
         * \param[in] model_file Path of model file, e.g ./yolov5face.onnx
         * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
         */
        explicit LprRecognizer(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption());

        [[nodiscard]] std::string name() const override { return "car-plate-recognizer"; }

        /** \brief Predict the face detection result for an input image
         *
         * \param[in] image The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
         * \param[in] result The output face detection result will be writen to this structure
         * \return true if the prediction successed, otherwise false
         */
        bool predict(cv::Mat& image, LprResult* result);

        /*! @brief
        Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default size = {640, 640}
        */
        std::vector<int> size{168, 48};
        std::vector<std::string> plate_color_list = {"黑色", "蓝色", "绿色", "白色", "黄色"};
        std::string plate_chr[78] = {
            "#", "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
            "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川",
            "贵", "云", "藏", "陕", "甘", "青", "宁",
            "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航", "危", "0", "1",
            "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
            "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
            "Y", "Z", "险", "品"
        };

    private:
        bool initialize();

        bool preprocess(cv::Mat& mat, Tensor* outputs);

        bool postprocess(std::vector<Tensor>& infer_result, LprResult* result);

        [[nodiscard]] bool is_dynamic_input() const { return is_dynamic_input_; }

        bool is_dynamic_input_{};
    };
} // namespace modeldeploy::vision::facedet
