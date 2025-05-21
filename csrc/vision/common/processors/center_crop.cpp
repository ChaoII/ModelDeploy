#include "center_crop.h"
#include "csrc/core/md_log.h"

namespace modeldeploy::vision {
    bool CenterCrop::impl(cv::Mat* im) const {
        const int height = im->rows;
        const int width = im->cols;
        if (height < height_ || width < width_) {
            MD_LOG_ERROR << "[CenterCrop] Image size less than crop size" << std::endl;
            return false;
        }
        const int offset_x = (width - width_) / 2;
        const int offset_y = (height - height_) / 2;
        const cv::Rect crop_roi(offset_x, offset_y, width_, height_);
        const cv::Mat new_im = (*im)(crop_roi).clone();
        *im = new_im;
        return true;
    }

    bool CenterCrop::operator()(cv::Mat* mat) const {
        return impl(mat);
    }

    bool CenterCrop::apply(cv::Mat* mat, const int& width, const int& height) {
        const auto op = CenterCrop(width, height);
        return op(mat);
    }
}
