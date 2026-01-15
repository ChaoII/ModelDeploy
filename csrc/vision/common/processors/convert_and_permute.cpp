//
// Created by aichao on 2025/2/20.
//

#include "convert_and_permute.h"

#include <core/md_log.h>

namespace modeldeploy::vision {
    ConvertAndPermute::ConvertAndPermute(const std::vector<float>& alpha,
                                         const std::vector<float>& beta,
                                         const bool swap_rb) {
        if (alpha.size() != beta.size()) {
            MD_LOG_ERROR << "ConvertAndPermute: requires the size of alpha equal to the size of beta." << std::endl;
        }

        if (alpha.empty() || beta.empty()) {
            MD_LOG_ERROR << "ConvertAndPermute: requires the size of alpha and beta > 0.";
        }
        alpha_ = alpha;
        beta_ = beta;
        swap_rb_ = swap_rb;
    }

    bool ConvertAndPermute::impl(cv::Mat* mat) const {
        if (!mat || mat->empty()) {
            MD_LOG_ERROR << "Error: Input matrix is null or empty" << std::endl;
            return false;
        }
        const int origin_w = mat->cols;
        const int origin_h = mat->rows;
        const int channels = mat->channels();
        std::vector<cv::Mat> split_im;
        cv::split(*mat, split_im);
        if (swap_rb_) std::swap(split_im[0], split_im[2]);
        for (int i = 0; i < channels; i++) {
            split_im[i].convertTo(split_im[i], CV_32FC1, alpha_[i], beta_[i]);
        }
        cv::Mat res(origin_h, origin_w, CV_32FC(channels));
        for (int i = 0; i < channels; ++i) {
            cv::extractChannel(split_im[i], cv::Mat(origin_h, origin_w, CV_32FC1,
                                                    res.ptr() + i * origin_h * origin_w * res.elemSize1()), 0);
        }
        *mat = res;
        return true;
    }

    bool ConvertAndPermute::operator()(cv::Mat* mat) const {
        return impl(mat);
    }

    bool ConvertAndPermute::apply(cv::Mat* mat, const std::vector<float>& alpha,
                                  const std::vector<float>& beta, const bool swap_rb) {
        const auto op = ConvertAndPermute(alpha, beta, swap_rb);
        return op(mat);
    }
}
