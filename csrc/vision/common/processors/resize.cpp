//
// Created by aichao on 2025/2/20.
//

#include "resize.h"
namespace modeldeploy::vision {
bool Resize::ImplByOpenCV(cv::Mat* im) {
  int origin_w = im->cols;
  int origin_h = im->rows;

  if (width_ == origin_w && height_ == origin_h) {
    return true;
  }
  if (fabs(scale_w_ - 1.0) < 1e-06 && fabs(scale_h_ - 1.0) < 1e-06) {
    return true;
  }

  if (width_ > 0 && height_ > 0) {
    if (use_scale_) {
      float scale_w = width_ * 1.0 / origin_w;
      float scale_h = height_ * 1.0 / origin_h;
      cv::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
    } else {
      cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
    }
  } else if (scale_w_ > 0 && scale_h_ > 0) {
    cv::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
  } else {
    std::cerr << "Resize: the parameters must satisfy (width > 0 && height > 0) "
               "or (scale_w > 0 && scale_h > 0)."
            << std::endl;
    return false;
  }
  return true;
}
bool Resize::operator()(cv::Mat* mat) {
  return ImplByOpenCV(mat);
}

bool Resize::Run(cv::Mat* mat, int width, int height, float scale_w,
                 float scale_h, int interp, bool use_scale) {
  if (mat->rows == height && mat->cols == width) {
    return true;
  }
  auto r = Resize(width, height, scale_w, scale_h, interp, use_scale);
  return r(mat);
}

}
