//
// Created by aichao on 2025/2/20.
//
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
namespace modeldeploy::vision {
    class  Pad  {
    public:
        Pad(int top, int bottom, int left, int right,
            const std::vector<float>& value) {
            top_ = top;
            bottom_ = bottom;
            left_ = left;
            right_ = right;
            value_ = value;
        }
        bool ImplByOpenCV(cv::Mat* mat);

        std::string Name() { return "Pad"; }

        static bool Run(cv::Mat* mat, const int& top, const int& bottom, const int& left,
                        const int& right, const std::vector<float>& value);

        bool operator()(cv::Mat* mat) ;
        bool SetPaddingSize(int top, int bottom, int left, int right) {
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