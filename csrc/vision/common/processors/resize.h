//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    class MODELDEPLOY_CXX_EXPORT Resize {
    public:
        Resize(const int width, const int height, const float scale_w = -1.0, const float scale_h = -1.0,
               const int interp = 1, bool const use_scale = false) {
            width_ = width;
            height_ = height;
            scale_w_ = scale_w;
            scale_h_ = scale_h;
            interp_ = interp;
            use_scale_ = use_scale;
        }

        bool operator()(cv::Mat* mat) const;

        bool impl(cv::Mat* mat) const;

        std::string name() { return "Resize"; }

        static bool apply(cv::Mat* mat, int width, int height, float scale_w = -1.0,
                          float scale_h = -1.0, int interp = 1, bool use_scale = false);

        bool set_width_and_height(const int width, const int height) {
            width_ = width;
            height_ = height;
            return true;
        }

        std::tuple<int, int> get_width_and_height() {
            return std::make_tuple(width_, height_);
        }

    private:
        int width_;
        int height_;
        float scale_w_ = -1.0;
        float scale_h_ = -1.0;
        int interp_ = 1;
        bool use_scale_ = false;
    };
}
