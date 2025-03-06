//
// Created by aichao on 2025/2/20.
//
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace modeldeploy::vision {
  class  ConvertAndPermute  {
  public:
    ConvertAndPermute(const std::vector<float>& alpha = std::vector<float>(),
                      const std::vector<float>& beta = std::vector<float>(),
                      bool swap_rb = false);
    bool ImplByOpenCV(cv::Mat* mat);

    std::string Name() { return "ConvertAndPermute"; }

    static bool Run(cv::Mat* mat, const std::vector<float>& alpha,
                    const std::vector<float>& beta, bool swap_rb = false);

    std::vector<float> GetAlpha() const { return alpha_; }

    bool operator()(cv::Mat* mat) ;
    void SetAlpha(const std::vector<float>& alpha) {
      alpha_.clear();
      std::vector<float>().swap(alpha_);
      alpha_.assign(alpha.begin(), alpha.end());
    }

    std::vector<float> GetBeta() const { return beta_; }


    void SetBeta(const std::vector<float>& beta) {
      beta_.clear();
      std::vector<float>().swap(beta_);
      beta_.assign(beta.begin(), beta.end());
    }

    bool GetSwapRB() {
      return swap_rb_;
    }


    void SetSwapRB(bool swap_rb) {
      swap_rb_ = swap_rb;
    }

  private:
    std::vector<float> alpha_;
    std::vector<float> beta_;
    bool swap_rb_;
  };
}