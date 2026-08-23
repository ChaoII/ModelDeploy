#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"

namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT Annotator {
public:
    explicit Annotator(ImageData* frame) : frame_(frame) {}
    bool begin(cv::Mat* m);
    void rectangle(const Rect2f& b, const cv::Scalar& color, int thickness = 2);
    void text(const std::string& s, Point2f org, const cv::Scalar& color, double scale = 0.6);
    void line(Point2f a, Point2f b, const cv::Scalar& color, int thickness = 2);
    void circle(Point2f c, int r, const cv::Scalar& color, int thickness = 2);
    void fill_polygon(const std::vector<Point2f>& pts, const cv::Scalar& color, double alpha = 0.3);
private:
    ImageData* frame_;
};
MODELDEPLOY_CXX_EXPORT void draw_box_labels(const Detections& d, ImageData* frame,
                                            const std::unordered_map<int, std::string>& labels = {});
MODELDEPLOY_CXX_EXPORT void draw_traces(const std::vector<Point2f>& trace, ImageData* frame,
                                        const cv::Scalar& color, int thickness = 2);
} // namespace modeldeploy::vision::tool
