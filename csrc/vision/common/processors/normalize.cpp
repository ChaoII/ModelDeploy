//
// Created by aichao on 2025/2/21.
//


#include "normalize.h"

namespace modeldeploy {
namespace vision {
Normalize::Normalize(const std::vector<float>& mean,
                     const std::vector<float>& std, bool is_scale,
                     const std::vector<float>& min,
                     const std::vector<float>& max, bool swap_rb) {
  if(mean.size() != std.size()){
    std::cerr<<           "Normalize: requires the size of mean equal to the size of std."<<std::endl;
  }
  std::vector<double> mean_(mean.begin(), mean.end());
  std::vector<double> std_(std.begin(), std.end());
  std::vector<double> min_(mean.size(), 0.0);
  std::vector<double> max_(mean.size(), 255.0);
  if (min.size() != 0) {
    if(
        min.size() != mean.size()){
        std::cerr<<"Normalize: while min is defined, requires the size of min equal to "
        "the size of mean."<<std::endl;}
    min_.assign(min.begin(), min.end());
  }
  if (max.size() != 0) {
    if(
        min.size() != mean.size()){
        std::cerr<<"Normalize: while max is defined, requires the size of max equal to "
        "the size of mean."<<std::endl;}
    max_.assign(max.begin(), max.end());
  }
  for (auto c = 0; c < mean_.size(); ++c) {
    double alpha = 1.0;
    if (is_scale) {
      alpha /= (max_[c] - min_[c]);
    }
    double beta = -1.0 * (mean_[c] + min_[c] * alpha) / std_[c];
    alpha /= std_[c];
    alpha_.push_back(alpha);
    beta_.push_back(beta);
  }
  swap_rb_ = swap_rb;
}

  bool Normalize::operator()(cv::Mat* mat) {
  return ImplByOpenCV(mat);
}
bool Normalize::ImplByOpenCV(cv::Mat* im) {
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  if (swap_rb_) std::swap(split_im[0], split_im[2]);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::merge(split_im, *im);
  return true;
}



bool Normalize::Run(cv::Mat* mat, const std::vector<float>& mean,
                    const std::vector<float>& std, bool is_scale,
                    const std::vector<float>& min,
                    const std::vector<float>& max ,bool swap_rb) {
  auto n = Normalize(mean, std, is_scale, min, max, swap_rb);
  return n(mat);
}

}  // namespace vision
}  // namespace fastdeploy
