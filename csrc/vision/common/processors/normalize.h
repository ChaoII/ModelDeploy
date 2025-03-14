//
// Created by aichao on 2025/2/21.
//


#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"
#include "csrc/core/md_tensor.h"

namespace modeldeploy {
namespace vision {
/*! @brief Processor for Normalize images with given paramters.
 */
class MODELDEPLOY_CXX_EXPORT Normalize {
 public:
  Normalize(const std::vector<float>& mean, const std::vector<float>& std,
            bool is_scale = true,
            const std::vector<float>& min = std::vector<float>(),
            const std::vector<float>& max = std::vector<float>(),
            bool swap_rb = false);
  bool ImplByOpenCV(cv::Mat* mat);
    bool operator()(cv::Mat* mat) ;

  std::string Name() { return "Normalize"; }

  // While use normalize, it is more recommend not use this function
  // this function will need to compute result = ((mat / 255) - mean) / std
  // if we use the following method
  // ```
  // auto norm = Normalize(...)
  // norm(mat)
  // ```
  // There will be some precomputation in contruct function
  // and the `norm(mat)` only need to compute result = mat * alpha + beta
  // which will reduce lots of time
  /** \brief Process the input images
   *
   * \param[in] mat The input image data, `result = mat * alpha + beta`
   * \param[in] mean target mean vector of output images
   * \param[in] std target std vector of output images
   * \param[in] max max value vector to be in target image
   * \param[in] min min value vector to be in target image
   * \param[in] lib to define OpenCV or FlyCV or CVCUDA will be used.
   * \param[in] swap_rb to define whether to swap r and b channel order
   * \return true if the process successed, otherwise false
   */
  static bool Run(cv::Mat* mat, const std::vector<float>& mean,
                  const std::vector<float>& std, bool is_scale = true,
                  const std::vector<float>& min = std::vector<float>(),
                  const std::vector<float>& max = std::vector<float>(),bool swap_rb = false);

  std::vector<float> GetAlpha() const { return alpha_; }
  std::vector<float> GetBeta() const { return beta_; }

  bool GetSwapRB() {
    return swap_rb_;
  }

  /** \brief Process the input images
   *
   * \param[in] swap_rb set the value of the swap_rb parameter
   */
  void SetSwapRB(bool swap_rb) {
    swap_rb_ = swap_rb;
  }

 private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
  MDTensor gpu_alpha_;
  MDTensor gpu_beta_;
  bool swap_rb_;
};
}  // namespace vision
}  // namespace fastdeploy
