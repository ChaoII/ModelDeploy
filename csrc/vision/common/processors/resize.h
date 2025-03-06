//
// Created by aichao on 2025/2/20.
//

#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <csrc/vision/detection/preprocessor.h>

namespace modeldeploy::vision {
    class Resize {
    public:
        Resize(int width, int height, float scale_w = -1.0, float scale_h = -1.0,
               int interp = 1, bool use_scale = false) {
            width_ = width;
            height_ = height;
            scale_w_ = scale_w;
            scale_h_ = scale_h;
            interp_ = interp;
            use_scale_ = use_scale;
        }

        bool operator()(cv::Mat* mat);


        bool ImplByOpenCV(cv::Mat* mat);
        std::string Name() { return "Resize"; }

        static bool Run(cv::Mat* mat, int width, int height, float scale_w = -1.0,
                        float scale_h = -1.0, int interp = 1, bool use_scale = false);

        bool SetWidthAndHeight(int width, int height) {
            width_ = width;
            height_ = height;
            return true;
        }

        std::tuple<int, int> GetWidthAndHeight() {
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
