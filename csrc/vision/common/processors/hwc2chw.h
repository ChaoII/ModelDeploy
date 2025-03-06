
//
// Created by aichao on 2025/2/21.
//


#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "csrc/core/md_tensor.h"

namespace modeldeploy {
namespace vision {

/*! @brief Processor for transform images from HWC to CHW.
 */
class  HWC2CHW  {
 public:
  bool ImplByOpenCV(cv::Mat* mat);

  std::string Name() { return "HWC2CHW"; }
    bool operator()(cv::Mat* mat) ;

  /** \brief Process the input images
   *
   * \param[in] mat The input image data
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \return true if the process successed, otherwise false
   */
  static bool Run(cv::Mat* mat);

};
}  // namespace vision
}  // namespace fastdeploy
