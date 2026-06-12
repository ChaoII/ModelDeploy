#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "config.hpp"
#include "inference_engine.hpp"

/// 绘制引擎：在图像上绘制检测结果
class DrawEngine {
public:
    explicit DrawEngine(const DrawConfig& cfg);
    ~DrawEngine() = default;

    /// 在 BGR 图像上绘制所有模型的结果
    void draw(cv::Mat& bgr_image,
              const std::vector<InferResult>& results);

    /// 获取随机颜色（按 label_id 缓存）
    static cv::Scalar get_color(int label_id);

private:
    DrawConfig cfg_;
    std::vector<cv::Scalar> colors_;

    void draw_detection(cv::Mat& image, const InferResult& result);
};
