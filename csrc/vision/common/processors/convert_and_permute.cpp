//
// Created by aichao on 2025/2/20.
//

#include "convert_and_permute.h"

#include <csrc/core/md_log.h>

namespace modeldeploy::vision {
    ConvertAndPermute::ConvertAndPermute(const std::vector<float>& alpha,
                                         const std::vector<float>& beta,
                                         const bool swap_rb) {
        if (alpha.size() != beta.size()) {
            MD_LOG_ERROR << "ConvertAndPermute: requires the size of alpha equal to the size of beta." << std::endl;
        }

        if (!(!alpha.empty() && !beta.empty())) {
            MD_LOG_ERROR << "ConvertAndPermute: requires the size of alpha and beta > 0.";
        }
        alpha_.assign(alpha.begin(), alpha.end());
        beta_.assign(beta.begin(), beta.end());
        swap_rb_ = swap_rb;
    }

    bool ConvertAndPermute::impl(cv::Mat* im) const {
        const int origin_w = im->cols;
        const int origin_h = im->rows;
        std::vector<cv::Mat> split_im;
        cv::split(*im, split_im);
        if (swap_rb_) std::swap(split_im[0], split_im[2]);
        for (int c = 0; c < im->channels(); c++) {
            split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
        }
        cv::Mat res(origin_h, origin_w, CV_32FC(im->channels()));
        for (int i = 0; i < im->channels(); ++i) {
            cv::extractChannel(split_im[i], cv::Mat(origin_h, origin_w, CV_32FC1,
                                                    res.ptr() + i * origin_h * origin_w * 4), 0);
        }
        *im = res;
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
