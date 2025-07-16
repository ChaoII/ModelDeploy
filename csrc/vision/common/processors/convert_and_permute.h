//
// Created by aichao on 2025/2/20.
//
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "core/md_decl.h"

namespace modeldeploy::vision {
    class MODELDEPLOY_CXX_EXPORT ConvertAndPermute {
    public:
        ConvertAndPermute(const std::vector<float>& alpha = std::vector<float>(),
                          const std::vector<float>& beta = std::vector<float>(),
                          bool swap_rb = false);
        bool impl(cv::Mat* mat) const;

        std::string name() { return "ConvertAndPermute"; }

        static bool apply(cv::Mat* mat, const std::vector<float>& alpha,
                          const std::vector<float>& beta, bool swap_rb = false);

        std::vector<float> get_alpha() const { return alpha_; }

        bool operator()(cv::Mat* mat) const;

        void set_alpha(const std::vector<float>& alpha) {
            alpha_.clear();
            std::vector<float>().swap(alpha_);
            alpha_.assign(alpha.begin(), alpha.end());
        }

        std::vector<float> get_beta() const { return beta_; }


        void set_beta(const std::vector<float>& beta) {
            beta_.clear();
            std::vector<float>().swap(beta_);
            beta_.assign(beta.begin(), beta.end());
        }

        bool get_swap_rb() {
            return swap_rb_;
        }


        void set_swap_rb(bool swap_rb) {
            swap_rb_ = swap_rb;
        }

    private:
        std::vector<float> alpha_;
        std::vector<float> beta_;
        bool swap_rb_;
    };
}
