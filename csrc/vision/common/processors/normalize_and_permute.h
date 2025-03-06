
//
// Created by aichao on 2025/2/21.
//

#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "csrc/core/md_tensor.h"


namespace modeldeploy::vision {

/*! @brief Processor for Normalize and Permute images from HWC to CHW.
 */
class  NormalizeAndPermute {
 public:
  NormalizeAndPermute(const std::vector<float>& mean,
                      const std::vector<float>& std, bool is_scale = true,
                      const std::vector<float>& min = std::vector<float>(),
                      const std::vector<float>& max = std::vector<float>(),
                      bool swap_rb = false);
  bool ImplByOpenCV(cv::Mat* mat);
  bool operator()(cv::Mat* mat) ;

  std::string Name() { return "NormalizeAndPermute"; }

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
   * \param[in] swap_rb to define whether to swap r and b channel order
   * \return true if the process successed, otherwise false
   */
  static bool Run(cv::Mat* mat, const std::vector<float>& mean,
                  const std::vector<float>& std, bool is_scale = true,
                  const std::vector<float>& min = std::vector<float>(),
                  const std::vector<float>& max = std::vector<float>(), bool swap_rb = false);

  /** \brief Process the input images
   *
   * \param[in] alpha set the value of the alpha parameter
   */
  void SetAlpha(const std::vector<float>& alpha) {
    alpha_.clear();
    std::vector<float>().swap(alpha_);
    alpha_.assign(alpha.begin(), alpha.end());
  }

  /** \brief Process the input images
   *
   * \param[in] beta set the value of the beta parameter
   */
  void SetBeta(const std::vector<float>& beta) {
    beta_.clear();
    std::vector<float>().swap(beta_);
    beta_.assign(beta.begin(), beta.end());
  }

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
}

