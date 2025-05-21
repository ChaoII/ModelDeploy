//
// Created by aichao on 2025/2/21.
//


#include "hwc2chw.h"

#include "csrc/vision/utils.h"

namespace modeldeploy::vision {
    bool HWC2CHW::impl(cv::Mat* im) {
        cv::Mat im_clone = im->clone();
        const int rh = im->rows;
        const int rw = im->cols;
        const int rc = im->channels();

        for (int i = 0; i < rc; ++i) {
            cv::extractChannel(
                im_clone,
                cv::Mat(rh, rw, im->type() % 8,
                        im->ptr() + i * rh * rw * Tensor::get_element_size(utils::cv_dtype_to_md_dtype(im->type()))),
                i);
        }
        return true;
    }

    bool HWC2CHW::operator()(cv::Mat* mat) {
        return impl(mat);
    }

    bool HWC2CHW::apply(cv::Mat* mat) {
        auto op = HWC2CHW();
        return op(mat);
    }
}
