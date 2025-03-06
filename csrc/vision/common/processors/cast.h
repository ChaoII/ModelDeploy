
//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_tensor.h"


namespace modeldeploy {
namespace vision {

/*! @brief Processor for cast images with given type deafault is float.
 */
class  Cast  {
 public:
  explicit Cast(const std::string& dtype = "float") : dtype_(dtype) {}
  bool ImplByOpenCV(cv::Mat* mat);


  bool operator()(cv::Mat* mat) ;


  std::string Name() { return "Cast"; }
  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] dtype type of data will be casted to
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(cv::Mat* mat, const std::string& dtype);

  std::string GetDtype() const { return dtype_; }

 private:
  std::string dtype_;

};
}  // namespace vision
}  // namespace fastdeploy
