

#include "vision/common/processors/convert.h"
#include "core/md_log.h"

namespace modeldeploy::vision {
    Convert::Convert(const std::vector<float>& alpha,
                     const std::vector<float>& beta) {
        if (alpha.size() != beta.size()) {
            MD_LOG_ERROR << "Convert: requires the size of alpha equal to the size of beta." << std::endl;
        }
        if (alpha.size() == 0) {
            MD_LOG_ERROR << "Convert: requires the size of alpha and beta > 0." << std::endl;
        }
        alpha_.assign(alpha.begin(), alpha.end());
        beta_.assign(beta.begin(), beta.end());
    }

    bool Convert::impl(cv::Mat* im) const {
        std::vector<cv::Mat> split_im;
        cv::split(*im, split_im);
        for (int c = 0; c < im->channels(); c++) {
            split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
        }
        cv::merge(split_im, *im);
        return true;
    }

    bool Convert::operator()(cv::Mat* mat) {
        return impl(mat);
    }

    bool Convert::apply(cv::Mat* mat, const std::vector<float>& alpha,
                        const std::vector<float>& beta) {
        auto op = Convert(alpha, beta);
        return op(mat);
    }
}
