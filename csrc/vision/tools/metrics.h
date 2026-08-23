#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT MetricsCounts { int tp{0}; int fp{0}; int fn{0}; };
struct MODELDEPLOY_CXX_EXPORT MetricsScores { double precision{0}; double recall{0}; double f1{0}; double map50{0}; };
MODELDEPLOY_CXX_EXPORT MetricsCounts count_tp_fp_fn(const std::vector<Rect2f>& preds,
                                                    const std::vector<float>& pred_scores,
                                                    const std::vector<Rect2f>& gt,
                                                    double iou_threshold = 0.5);
MODELDEPLOY_CXX_EXPORT MetricsScores evaluate_metrics(const std::vector<Rect2f>& preds,
                                                      const std::vector<float>& pred_scores,
                                                      const std::vector<Rect2f>& gt,
                                                      double iou_threshold = 0.5);
} // namespace modeldeploy::vision::tool
