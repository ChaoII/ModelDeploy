#include "center_crop.h"

namespace modeldeploy::vision {
    bool CenterCrop::ImplByOpenCV(cv::Mat* im) {
        int height = static_cast<int>(im->rows);
        int width = static_cast<int>(im->cols);
        if (height < height_ || width < width_) {
            std::cerr << "[CenterCrop] Image size less than crop size" << std::endl;
            return false;
        }
        int offset_x = static_cast<int>((width - width_) / 2);
        int offset_y = static_cast<int>((height - height_) / 2);
        cv::Rect crop_roi(offset_x, offset_y, width_, height_);
        cv::Mat new_im = (*im)(crop_roi).clone();
        *im = new_im;
        return true;
    }

    bool CenterCrop::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }

    bool CenterCrop::Run(cv::Mat* mat, const int& width, const int& height) {
        auto c = CenterCrop(width, height);
        return c(mat);
    }
}
