#include "color_space_convert.h"


namespace modeldeploy::vision {
    bool BGR2RGB::ImplByOpenCV(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_BGR2RGB);
        *im = new_im;
        return true;
    }

    bool RGB2BGR::ImplByOpenCV(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_RGB2BGR);
        *im = new_im;
        return true;
    }

    bool BGR2GRAY::ImplByOpenCV(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_BGR2GRAY);
        *im = new_im;
        return true;
    }

    bool RGB2GRAY::ImplByOpenCV(cv::Mat* im) {
        cv::Mat new_im;
        cv::cvtColor(*im, new_im, cv::COLOR_RGB2GRAY);
        *im = new_im;
        return true;
    }

    bool BGR2RGB::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }

    bool RGB2BGR::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }

    bool BGR2GRAY::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }

    bool RGB2GRAY::operator()(cv::Mat* mat) {
        return ImplByOpenCV(mat);
    }


    bool BGR2RGB::Run(cv::Mat* mat) {
        auto b = BGR2RGB();
        return b(mat);
    }

    bool RGB2BGR::Run(cv::Mat* mat) {
        auto r = RGB2BGR();
        return r(mat);
    }

    bool BGR2GRAY::Run(cv::Mat* mat) {
        auto b = BGR2GRAY();
        return b(mat);
    }

    bool RGB2GRAY::Run(cv::Mat* mat) {
        auto r = RGB2GRAY();
        return r(mat);
    }
} // namespace modeldeploy::vision
