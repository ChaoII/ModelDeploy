//
// Created by aichao on 2025/2/20.
//

#include "resize.h"

#include <csrc/core/md_log.h>

namespace modeldeploy::vision {
    bool Resize::impl(cv::Mat* im) const {
        const int origin_w = im->cols;
        const int origin_h = im->rows;

        if (width_ == origin_w && height_ == origin_h) {
            return true;
        }
        if (fabs(scale_w_ - 1.0) < 1e-06 && fabs(scale_h_ - 1.0) < 1e-06) {
            return true;
        }

        if (width_ > 0 && height_ > 0) {
            if (use_scale_) {
                const float scale_w = width_ * 1.0 / origin_w;
                const float scale_h = height_ * 1.0 / origin_h;
                cv::resize(*im, *im, cv::Size(0, 0), scale_w, scale_h, interp_);
            }
            else {
                cv::resize(*im, *im, cv::Size(width_, height_), 0, 0, interp_);
            }
        }
        else if (scale_w_ > 0 && scale_h_ > 0) {
            cv::resize(*im, *im, cv::Size(0, 0), scale_w_, scale_h_, interp_);
        }
        else {
            MD_LOG_ERROR << "Resize: the parameters must satisfy (width > 0 && height > 0) "
                "or (scale_w > 0 && scale_h > 0)." << std::endl;
            return false;
        }
        return true;
    }

    bool Resize::operator()(cv::Mat* mat) const {
        return impl(mat);
    }

    bool Resize::apply(cv::Mat* mat, const int width, const int height, const float scale_w,
                       const float scale_h, const int interp, const bool use_scale) {
        if (mat->rows == height && mat->cols == width) {
            return true;
        }
        const auto op = Resize(width, height, scale_w, scale_h, interp, use_scale);
        return op(mat);
    }
}
