//
// Created by aichao on 2025/2/20.
//
#include "pad.h"
namespace modeldeploy::vision {
  bool Pad::ImplByOpenCV(cv::Mat* im) {

    if (im->channels() > 4) {
      std::cerr << "Pad: Only support channels <= 4." << std::endl;
      return false;
    }
    if (im->channels() != value_.size()) {
      std::cerr  << "Pad: Require input channels equals to size of padding value, "
                 "but now channels = "
              << im->channels()
              << ", the size of padding values = " << value_.size() << "."
              << std::endl;
      return false;
    }

    cv::Scalar value;
    if (value_.size() == 1) {
      value = cv::Scalar(value_[0]);
    } else if (value_.size() == 2) {
      value = cv::Scalar(value_[0], value_[1]);
    } else if (value_.size() == 3) {
      value = cv::Scalar(value_[0], value_[1], value_[2]);
    } else {
      value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
    }
    cv::copyMakeBorder(*im, *im, top_, bottom_, left_, right_,cv::BORDER_CONSTANT, value);
    return true;
  }

  bool Pad::operator()(cv::Mat* mat) {
    return ImplByOpenCV(mat);
  }


  bool Pad::Run(cv::Mat* mat, const int& top, const int& bottom, const int& left,
                const int& right, const std::vector<float>& value) {
    auto p = Pad(top, bottom, left, right, value);
    return p(mat);
  }
}