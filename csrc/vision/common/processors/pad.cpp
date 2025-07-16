//
// Created by aichao on 2025/2/20.
//

#include "core/md_log.h"
#include "vision/common/processors/pad.h"

namespace modeldeploy::vision {
    bool Pad::impl(cv::Mat* im) const {
        if (im->channels() > 4) {
            MD_LOG_ERROR << "Pad: Only support channels <= 4." << std::endl;
            return false;
        }
        if (im->channels() != value_.size()) {
            MD_LOG_ERROR << "Pad: Require input channels equals to size of padding value, "
                "but now channels = "
                << im->channels()
                << ", the size of padding values = " << value_.size() << "."
                << std::endl;
            return false;
        }

        cv::Scalar value;
        if (value_.size() == 1) {
            value = cv::Scalar(value_[0]);
        }
        else if (value_.size() == 2) {
            value = cv::Scalar(value_[0], value_[1]);
        }
        else if (value_.size() == 3) {
            value = cv::Scalar(value_[0], value_[1], value_[2]);
        }
        else {
            value = cv::Scalar(value_[0], value_[1], value_[2], value_[3]);
        }
        cv::copyMakeBorder(*im, *im, top_, bottom_, left_, right_, cv::BORDER_CONSTANT, value);
        return true;
    }

    bool Pad::operator()(cv::Mat* mat) const {
        return impl(mat);
    }


    bool Pad::apply(cv::Mat* mat, const int& top, const int& bottom, const int& left,
                    const int& right, const std::vector<float>& value) {
        const auto op = Pad(top, bottom, left, right, value);
        return op(mat);
    }
}
