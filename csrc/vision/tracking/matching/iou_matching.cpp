#include "vision/tracking/matching/iou_matching.h"

#include <algorithm>

namespace modeldeploy::vision::tracking {
    float iou(const Rect2f& a, const Rect2f& b) {
        const float ax1 = a.x, ay1 = a.y;
        const float ax2 = a.x + a.width, ay2 = a.y + a.height;
        const float bx1 = b.x, by1 = b.y;
        const float bx2 = b.x + b.width, by2 = b.y + b.height;

        const float xx1 = std::max(ax1, bx1), yy1 = std::max(ay1, by1);
        const float xx2 = std::min(ax2, bx2), yy2 = std::min(ay2, by2);

        const float w = std::max(0.0f, xx2 - xx1);
        const float h = std::max(0.0f, yy2 - yy1);
        const float inter = w * h;

        const float area_a = a.width * a.height;
        const float area_b = b.width * b.height;
        const float uni = area_a + area_b - inter;
        if (uni <= 0.0f) return 0.0f;
        return inter / uni;
    }

    std::vector<std::vector<float>> iou_distance(const std::vector<Rect2f>& a, const std::vector<Rect2f>& b) {
        std::vector<std::vector<float>> dist(a.size(), std::vector<float>(b.size(), 1.0f));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                dist[i][j] = 1.0f - iou(a[i], b[j]);
            }
        }
        return dist;
    }
}
