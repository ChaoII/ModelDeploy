// ModelDeploy demo_tools —— 纯工具演示（无权重）：Annotator/Zone/Metrics/Slicer。
// Usage: demo_tools <out.png>
//   读一张图（或合成画布），演示 draw_box_labels + LineZone + PolygonZone + Metrics。
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vision/common/image_data.h"
#include "vision/tools/annotator.h"
#include "vision/tools/detections.h"
#include "vision/tools/metrics.h"
#include "vision/tools/zone.h"

using namespace modeldeploy::vision;

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: demo_tools <out.png>\n"); return 1; }
    const std::string outPath = argv[1];

    // 合成一张 320x240 画布（无权重也能跑）。
    cv::Mat canvas(240, 320, CV_8UC3, cv::Scalar(240, 240, 240));
    ImageData frame(canvas);

    // Annotator + draw_box_labels 演示。
    tool::Detections d;
    d.boxes      = {Rect2f(10, 10, 60, 40), Rect2f(120, 60, 50, 70)};
    d.class_id   = {0, 1};
    d.confidence = {0.92f, 0.85f};
    tool::draw_box_labels(d, &frame, {{0, "person"}, {1, "car"}});

    // LineZone 跨线计数演示。
    tool::LineZone zone(Point2f(160, 0), Point2f(160, 240));
    zone.trigger(Point2f(100, 120));
    zone.trigger(Point2f(200, 120));
    printf("line_zone.trigger_count = %d\n", zone.trigger_count());

    // PolygonZone 进入计数演示。
    tool::PolygonZone pz({Point2f(0, 0), Point2f(320, 0), Point2f(320, 240), Point2f(0, 240)});
    pz.update({Point2f(50, 50), Point2f(300, 200), Point2f(159, 159)});
    printf("polygon_zone.current_count = %d, contains(10,10)=%d\n",
           pz.current_count(), (int)pz.contains(Point2f(10, 10)));

    // Metrics 小样本 mAP/precision/recall/f1 演示。
    std::vector<Rect2f> preds = {Rect2f(0, 0, 10, 10), Rect2f(5, 5, 10, 10), Rect2f(50, 50, 5, 5)};
    std::vector<float> scores = {0.9f, 0.8f, 0.7f};
    std::vector<Rect2f> gt    = {Rect2f(0, 0, 10, 10), Rect2f(5, 5, 10, 10)};
    auto m = tool::evaluate_metrics(preds, scores, gt, 0.5);
    printf("metrics: map50=%.3f precision=%.3f recall=%.3f f1=%.3f\n",
           m.map50, m.precision, m.recall, m.f1);

    cv::imwrite(outPath, canvas);
    printf("wrote %s\n", outPath.c_str());
    return 0;
}
