//
// Created by aichao on 2025/2/21.
//


#include "hwc2chw.h"

#include "csrc/vision/utils.h"

namespace modeldeploy::vision {
    bool HWC2CHW::ImplByOpenCV(cv::Mat* im) {
        cv::Mat im_clone = im->clone();
        int rh = im->rows;
        int rw = im->cols;
        int rc = im->channels();

        for (int i = 0; i < rc; ++i) {
            cv::extractChannel(
                im_clone,
                cv::Mat(rh, rw, im->type() % 8,
                        im->ptr() + i * rh * rw * MDDataType::size(utils::cv_dtype_to_md_dtype(im->type()))), i);
        }
        return true;
    }

    bool HWC2CHW::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }


    bool HWC2CHW::Run(cv::Mat* mat) {
        auto h = HWC2CHW();
        return h(mat);
    }
}
