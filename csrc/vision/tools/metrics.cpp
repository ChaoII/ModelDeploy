#include "vision/tools/metrics.h"
#include "vision/tools/detections.h"
#include <algorithm>
#include <utility>
namespace modeldeploy::vision::tool {
MetricsCounts count_tp_fp_fn(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores,
                             const std::vector<Rect2f>& gt, double iou_threshold) {
    std::vector<size_t> order(preds.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return pred_scores[a] > pred_scores[b]; });
    std::vector<bool> gt_matched(gt.size(), false);
    MetricsCounts c;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const auto& p = preds[order[oi]];
        int best = -1; double best_iou = iou_threshold;
        for (size_t g = 0; g < gt.size(); ++g) {
            if (gt_matched[g]) continue;
            double v = iou(p, gt[g]);
            if (v >= best_iou) { best_iou = v; best = (int)g; }
        }
        if (best >= 0) { gt_matched[(size_t)best] = true; ++c.tp; } else ++c.fp;
    }
    c.fn = (int)gt.size() - c.tp;
    return c;
}
MetricsScores evaluate_metrics(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores,
                               const std::vector<Rect2f>& gt, double iou_threshold) {
    auto mc = count_tp_fp_fn(preds, pred_scores, gt, iou_threshold);
    MetricsScores s;
    const int denom = mc.tp + mc.fp;
    s.precision = denom > 0 ? (double)mc.tp / denom : 0.0;
    const int gtdenom = mc.tp + mc.fn;
    s.recall = gtdenom > 0 ? (double)mc.tp / gtdenom : 0.0;
    s.f1 = (s.precision + s.recall) > 0 ? 2.0 * s.precision * s.recall / (s.precision + s.recall) : 0.0;
    // 11 点插值 mAP（按置信度降序扫描）
    std::vector<size_t> order(preds.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return pred_scores[a] > pred_scores[b]; });
    std::vector<bool> gt_matched(gt.size(), false);
    std::vector<std::pair<double,double>> pr;
    int tp = 0, count = 0;
    if (gt.empty()) pr.emplace_back(0.0, 0.0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const auto& p = preds[order[oi]];
        int best = -1; double best_iou = iou_threshold;
        for (size_t g = 0; g < gt.size(); ++g) {
            if (gt_matched[g]) continue;
            double v = iou(p, gt[g]);
            if (v >= best_iou) { best_iou = v; best = (int)g; }
        }
        ++count;
        if (best >= 0) { gt_matched[(size_t)best] = true; ++tp; }
        const double recall_val = gt.empty() ? 0.0 : (double)tp / gt.size();
        pr.emplace_back(recall_val, (double)tp / count);
    }
    double ap = 0.0;
    for (int r = 0; r <= 10; ++r) {
        const double target = r / 10.0;
        double maxp = 0.0;
        for (const auto& e : pr) if (e.first >= target) maxp = std::max(maxp, e.second);
        ap += maxp / 11.0;
    }
    s.map50 = ap;
    return s;
}
} // namespace modeldeploy::vision::tool
