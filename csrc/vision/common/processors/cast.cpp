//
// Created by aichao on 2025/2/21.
//


#include "cast.h"


namespace modeldeploy::vision {
    bool Cast::ImplByOpenCV(cv::Mat* mat) {
        int c = mat->channels();
        if (dtype_ == "float") {
            if (mat->type() != CV_32FC(c)) {
                mat->convertTo(*mat, CV_32FC(c));
            }
        }
        else if (dtype_ == "float16") {
            if (mat->type() != CV_16FC(c)) {
                mat->convertTo(*mat, CV_16FC(c));
            }
        }
        else if (dtype_ == "double") {
            if (mat->type() != CV_64FC(c)) {
                mat->convertTo(*mat, CV_64FC(c));
            }
        }
        else {
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
} // namespace modeldeploy::vision
