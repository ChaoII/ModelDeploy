

#pragma once
#include "csrc/base_model.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy {

namespace vision {

namespace facedet {
/*! @brief YOLOv5Face model object used when to load a YOLOv5Face model exported by YOLOv5Face.
 */
class MODELDEPLOY_CXX_EXPORT CarplateRecognizer : public BaseModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./yolov5face.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  CarplateRecognizer(const std::string& model_file,const RuntimeOption& custom_option = RuntimeOption());

  std::string ModelName() const { return "yolov5-face"; }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output face detection result will be writen to this structure
   * \param[in] conf_threshold confidence threashold for postprocessing, default is 0.25
   * \param[in] nms_iou_threshold iou threashold for NMS, default is 0.5
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, FaceDetectionResult* result,
                       float conf_threshold = 0.25,
                       float nms_iou_threshold = 0.5);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the target size after resize, default size = {640, 640}
  */
  std::vector<int> size{168,48};
    std::vector<std::string>plate_color_list={"黑色","蓝色","绿色","白色","黄色"};
    std::string plate_chr[78]={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁",
    "新","学","警","港","澳","挂","使","领","民","航","危","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品"};

 private:
  bool Initialize();

  bool Preprocess(cv::Mat* mat, MDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  bool Postprocess(std::vector<MDTensor>& infer_result, FaceDetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  bool is_dynamic_input_;
};

}  // namespace facedet
}  // namespace vision
}  // namespace fastdeploy
