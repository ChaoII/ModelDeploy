//
// Created by aichao on 2025/2/21.
//


#include "cast.h"
#include "core/md_log.h"
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision {
    bool Cast::impl(cv::Mat* mat) const {
        const int c = mat->channels();
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
            MD_LOG_ERROR << "Cast not support for " << dtype_ << " now! will skip this operation." << std::endl;
            return false;
        }
        return true;
    }

    bool Cast::operator()(cv::Mat* mat) const {
        return impl(mat);
    }


    bool Cast::apply(cv::Mat* mat, const std::string& dtype) {
        const auto op = Cast(dtype);
        return op(mat);
    }
} // namespace modeldeploy::vision
