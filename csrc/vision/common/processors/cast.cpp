
//
// Created by aichao on 2025/2/21.
//


#include "cast.h"


namespace modeldeploy {
namespace vision {

bool Cast::ImplByOpenCV(cv::Mat* im) {
  int c = im->channels();
  if (dtype_ == "float") {
    if (im->type() != CV_32FC(c)) {
      im->convertTo(*im, CV_16FC(c));
    }
  } else if (dtype_ == "double") {
    if (im->type() != CV_64FC(c)) {
      im->convertTo(*im, CV_64FC(c));
    }
  } else {
    std::cerr << "Cast not support for " << dtype_
              << " now! will skip this operation." << std::endl;
  }
  return true;
}
  bool Cast::operator()(cv::Mat* mat) {
  return ImplByOpenCV(mat);
}


bool Cast::Run(cv::Mat* mat, const std::string& dtype) {
  auto c = Cast(dtype);
  return c(mat);
}

}  // namespace vision
}  // namespace fastdeploy
