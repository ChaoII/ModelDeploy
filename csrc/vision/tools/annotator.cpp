#include "vision/tools/annotator.h"
#include <sstream>
namespace modeldeploy::vision::tool {
bool Annotator::begin(cv::Mat* m) { return frame_ && frame_->asMat(m); }
void Annotator::rectangle(const Rect2f& b, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::rectangle(m, cv::Rect((int)b.x, (int)b.y, (int)b.width, (int)b.height), color, thickness);
}
void Annotator::text(const std::string& s, Point2f org, const cv::Scalar& color, double scale) {
    cv::Mat m; if (!begin(&m)) return;
    cv::putText(m, s, cv::Point((int)org.x, (int)org.y), cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}
void Annotator::line(Point2f a, Point2f b, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::line(m, cv::Point((int)a.x, (int)a.y), cv::Point((int)b.x, (int)b.y), color, thickness);
}
void Annotator::circle(Point2f c, int r, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::circle(m, cv::Point((int)c.x, (int)c.y), r, color, thickness);
}
void Annotator::fill_polygon(const std::vector<Point2f>& pts, const cv::Scalar& color, double alpha) {
    cv::Mat m; if (!begin(&m)) return;
    std::vector<cv::Point> poly;
    for (const auto& p : pts) poly.emplace_back((int)p.x, (int)p.y);
    cv::Mat overlay = m.clone();
    cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{poly}, color);
    cv::addWeighted(overlay, alpha, m, 1.0 - alpha, 0.0, m);
}
void draw_box_labels(const Detections& d, ImageData* frame,
                     const std::unordered_map<int, std::string>& labels) {
    Annotator ann(frame);
    for (size_t i = 0; i < d.size(); ++i) {
        const auto& b = d.boxes[i];
        ann.rectangle(b, cv::Scalar(0, 0, 255), 2);
        std::ostringstream oss;
        oss << (labels.count(d.class_id[i]) ? labels.at(d.class_id[i]) : std::to_string(d.class_id[i]));
        oss << " " << d.confidence[i];
        ann.text(oss.str(), Point2f(b.x, std::max(0.0f, b.y - 4)), cv::Scalar(0, 255, 255), 0.5);
    }
}
void draw_traces(const std::vector<Point2f>& trace, ImageData* frame,
                 const cv::Scalar& color, int thickness) {
    Annotator ann(frame);
    for (size_t i = 1; i < trace.size(); ++i)
        ann.line(trace[i - 1], trace[i], color, thickness);
}
} // namespace modeldeploy::vision::tool
