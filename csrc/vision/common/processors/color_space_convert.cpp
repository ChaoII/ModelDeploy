#include "color_space_convert.h"


namespace modeldeploy::vision {
    bool BGR2RGB::impl(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_BGR2RGB);
        *im = new_im;
        return true;
    }

    bool RGB2BGR::impl(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
        *im = new_im;
        return true;
    }

    bool BGR2GRAY::impl(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_BGR2GRAY);
        *im = new_im;
        return true;
    }

    bool RGB2GRAY::impl(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_RGB2GRAY);
        *im = new_im;
        return true;
    }

    bool BGR2RGB::operator()(cv::Mat* mat) {
        return impl(mat);
    }

    bool RGB2BGR::operator()(cv::Mat* mat) {
        return impl(mat);
    }

    bool BGR2GRAY::operator()(cv::Mat* mat) {
        return impl(mat);
    }

    bool RGB2GRAY::operator()(cv::Mat* mat) {
        return impl(mat);
    }


    bool BGR2RGB::apply(cv::Mat* mat) {
        auto op = BGR2RGB();
        return op(mat);
    }

    bool RGB2BGR::apply(cv::Mat* mat) {
        auto op = RGB2BGR();
        return op(mat);
    }

    bool BGR2GRAY::apply(cv::Mat* mat) {
        auto op = BGR2GRAY();
        return op(mat);
    }

    bool RGB2GRAY::apply(cv::Mat* mat) {
        auto op = RGB2GRAY();
        return op(mat);
    }
} // namespace modeldeploy::vision
