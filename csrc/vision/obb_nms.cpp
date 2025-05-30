//
// Created by aichao on 2025/5/30.
//

#include <array>
#include <numeric>
#include <vector>
#include <ranges>
#include "csrc/vision/utils.h"

namespace modeldeploy::vision::utils {
    struct Point {
        float x, y;
    };

    void sort_detection_result_by_score(DetectionResult& result) {
        const size_t num = result.scores.size();
        std::vector<size_t> indices(num);
        std::iota(indices.begin(), indices.end(), 0); // 初始化索引 0,1,...,N-1

        // 按照 scores 降序排序索引
        std::ranges::sort(indices, [&](size_t i1, size_t i2) {
            return result.scores[i1] > result.scores[i2];
        });

        // 使用排序后的索引，重排所有字段
        auto reorder = [&](auto& vec) {
            using T = typename std::decay<decltype(vec[0])>::type;
            std::vector<T> reordered;
            reordered.reserve(vec.size());
            for (size_t idx : indices) {
                reordered.push_back(vec[idx]);
            }
            vec = std::move(reordered);
        };

        reorder(result.scores);
        reorder(result.label_ids);
        if (!result.boxes.empty()) reorder(result.boxes);
        if (!result.rotated_boxes.empty()) reorder(result.rotated_boxes);
        if (result.contain_masks && !result.masks.empty()) reorder(result.masks);
    }

    // Shoelace 算法求多边形面积（顶点顺时针/逆时针都可以）
    float polygon_area(const std::vector<Point>& poly) {
        float area = 0.0f;
        int n = poly.size();
        for (int i = 0; i < n; ++i) {
            const Point& p1 = poly[i];
            const Point& p2 = poly[(i + 1) % n];
            area += p1.x * p2.y - p2.x * p1.y;
        }
        return std::abs(area) * 0.5f;
    }

    // 判断点 p 是否在边 (a,b) 的左侧
    bool is_inside(const Point& p, const Point& a, const Point& b) {
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0;
    }

    // 计算两个边的交点
    Point compute_intersection(const Point& p1, const Point& p2, const Point& q1, const Point& q2) {
        float A1 = p2.y - p1.y;
        float B1 = p1.x - p2.x;
        float C1 = A1 * p1.x + B1 * p1.y;

        float A2 = q2.y - q1.y;
        float B2 = q1.x - q2.x;
        float C2 = A2 * q1.x + B2 * q1.y;

        float det = A1 * B2 - A2 * B1;
        if (std::abs(det) < 1e-6f) return {0, 0}; // 平行，实际使用中需处理

        return {
            (B2 * C1 - B1 * C2) / det,
            (A1 * C2 - A2 * C1) / det
        };
    }

    // Sutherland–Hodgman 多边形裁剪
    std::vector<Point> polygon_intersection(const std::vector<Point>& subject, const std::vector<Point>& clip) {
        std::vector<Point> output = subject;
        for (size_t i = 0; i < clip.size(); ++i) {
            const Point& A = clip[i];
            const Point& B = clip[(i + 1) % clip.size()];
            std::vector<Point> input = output;
            output.clear();

            for (size_t j = 0; j < input.size(); ++j) {
                const Point& P = input[j];
                const Point& Q = input[(j + 1) % input.size()];
                if (is_inside(Q, A, B)) {
                    if (!is_inside(P, A, B)) {
                        output.push_back(compute_intersection(P, Q, A, B));
                    }
                    output.push_back(Q);
                }
                else if (is_inside(P, A, B)) {
                    output.push_back(compute_intersection(P, Q, A, B));
                }
            }
        }
        return output;
    }

    // 转换 array<float, 8> 为 Point vector
    std::vector<Point> array_to_polygon(const std::array<float, 8>& box) {
        return {
            {box[0], box[1]},
            {box[2], box[3]},
            {box[4], box[5]},
            {box[6], box[7]},
        };
    }

    // 主函数：计算两个旋转框的 IoU
    float rotated_iou(const std::array<float, 8>& box1, const std::array<float, 8>& box2) {
        std::vector<Point> poly1 = array_to_polygon(box1);
        std::vector<Point> poly2 = array_to_polygon(box2);

        float area1 = polygon_area(poly1);
        float area2 = polygon_area(poly2);

        std::vector<Point> inter_poly = polygon_intersection(poly1, poly2);
        if (inter_poly.empty()) return 0.0f;

        float inter_area = polygon_area(inter_poly);
        float union_area = area1 + area2 - inter_area;
        return inter_area / union_area;
    }

    void obb_nms(DetectionResult* output, const float iou_threshold,
                 std::vector<int>* index) {
        // get sorted score indices
        std::vector<int> sorted_indices;
        if (index != nullptr) {
            std::map<float, int, std::greater<>> score_map;
            for (size_t i = 0; i < output->scores.size(); ++i) {
                score_map.insert({output->scores[i], static_cast<int>(i)});
            }
            for (auto val : score_map | std::views::values) {
                sorted_indices.push_back(val);
            }
        }
        sort_detection_result_by_score(*output);
        std::vector suppressed(output->rotated_boxes.size(), 0);
        for (size_t i = 0; i < output->rotated_boxes.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            for (size_t j = i + 1; j < output->rotated_boxes.size(); ++j) {
                if (suppressed[j] == 1) {
                    continue;
                }
                auto iou = rotated_iou(output->rotated_boxes[i], output->rotated_boxes[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
        DetectionResult backup(*output);
        output->clear();
        output->reserve(static_cast<int>(suppressed.size()));
        for (size_t i = 0; i < suppressed.size(); ++i) {
            if (suppressed[i] == 1) {
                continue;
            }
            output->rotated_boxes.emplace_back(backup.rotated_boxes[i]);
            output->scores.push_back(backup.scores[i]);
            output->label_ids.push_back(backup.label_ids[i]);
            if (index != nullptr) {
                index->push_back(sorted_indices[i]);
            }
        }
    }
}
