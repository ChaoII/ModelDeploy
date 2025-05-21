//
// Created by aichao on 2025/2/20.
//
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "csrc/core/md_decl.h"

namespace modeldeploy::vision {
    class MODELDEPLOY_CXX_EXPORT Pad {
    public:
        Pad(const int top, const int bottom, const int left, const int right,
            const std::vector<float>& value) {
            top_ = top;
            bottom_ = bottom;
            left_ = left;
            right_ = right;
            value_ = value;
        }

        bool impl(cv::Mat* mat) const;

        std::string name() { return "Pad"; }

        static bool apply(cv::Mat* mat, const int& top, const int& bottom, const int& left,
                          const int& right, const std::vector<float>& value);

        bool operator()(cv::Mat* mat) const;

        bool set_padding_size(const int top, const int bottom, const int left, const int right) {
            top_ = top;
            bottom_ = bottom;
            left_ = left;
            right_ = right;
            return true;
        }

    private:
        int top_;
        int bottom_;
        int left_;
        int right_;
        std::vector<float> value_;
    };
}
