#include "vision/solutions/object_blur.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
namespace modeldeploy::vision::solution {
void ObjectBlur::blur(const ImageData& img, const Rect2f& box, ImageData* out) const {
    if (!out) return;
    cv::Mat src;
    if (!img.asMat(&src)) return;
    cv::Mat dst = src.clone();
    const int x = (int)box.x, y = (int)box.y;
    const int w = (int)box.width, h = (int)box.height;
    cv::Rect roi(std::max(0, x), std::max(0, y),
                 std::min(w, src.cols - std::max(0, x)), std::min(h, src.rows - std::max(0, y)));
    if (roi.width > 0 && roi.height > 0) {
        cv::Mat region = dst(roi);
        cv::GaussianBlur(region, region, cv::Size(ksize_, ksize_), 0);
    }
    *out = ImageData(dst);
}
} // namespace modeldeploy::vision::solution
