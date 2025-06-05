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

    // Shoelace 算法求多边形面积（顶点顺时针/逆时针都可以）
    float polygon_area(const std::vector<cv::Point2f>& poly) {
        float area = 0.0f;
        const size_t n = poly.size();
        for (size_t i = 0; i < n; ++i) {
            const cv::Point2f& p1 = poly[i];
            const cv::Point2f& p2 = poly[(i + 1) % n];
            area += p1.x * p2.y - p2.x * p1.y;
        }
        return std::abs(area) * 0.5f;
    }

    // 判断点 p 是否在边 (a,b) 的左侧
    bool is_inside(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0;
    }

    // 计算两个边的交点
    cv::Point2f compute_intersection(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& q1,
                                     const cv::Point2f& q2) {
        const float A1 = p2.y - p1.y;
        const float B1 = p1.x - p2.x;
        const float C1 = A1 * p1.x + B1 * p1.y;

        const float A2 = q2.y - q1.y;
        const float B2 = q1.x - q2.x;
        const float C2 = A2 * q1.x + B2 * q1.y;

        const float det = A1 * B2 - A2 * B1;
        if (std::abs(det) < 1e-6f) return {0, 0}; // 平行，实际使用中需处理
        return {
            (B2 * C1 - B1 * C2) / det,
            (A1 * C2 - A2 * C1) / det
        };
    }

    // Sutherland–Hodgman 多边形裁剪
    std::vector<cv::Point2f> polygon_intersection(const std::vector<cv::Point2f>& subject,
                                                  const std::vector<cv::Point2f>& clip) {
        std::vector<cv::Point2f> output = subject;
        for (size_t i = 0; i < clip.size(); ++i) {
            const cv::Point2f& A = clip[i];
            const cv::Point2f& B = clip[(i + 1) % clip.size()];
            std::vector<cv::Point2f> input = output;
            output.clear();

            for (size_t j = 0; j < input.size(); ++j) {
                const cv::Point2f& P = input[j];
                const cv::Point2f& Q = input[(j + 1) % input.size()];
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
    std::vector<cv::Point2f> array_to_polygon(const std::array<float, 8>& box) {
        return {
            {box[0], box[1]},
            {box[2], box[3]},
            {box[4], box[5]},
            {box[6], box[7]},
        };
    }

    // 使用OpenCV中的多边形相交方法
    float rotated_iou(const cv::RotatedRect& box1, const cv::RotatedRect& box2) {
        std::vector<cv::Point2f> intersection;
        float iou = 0.0f;
        if (cv::rotatedRectangleIntersection(box1, box2, intersection) > 0) {
            const auto intersection_area = cv::contourArea(intersection);
            const auto box1_area = box1.size.area();
            const auto box2_area = box2.size.area();
            iou = static_cast<float>(intersection_area / (box1_area + box2_area - intersection_area));
        }
        return iou;
    }


    // 计算两个旋转框的 IoU，自己造轮子
    float rotated_iou(const std::array<float, 8>& box1, const std::array<float, 8>& box2) {
        const auto poly1 = array_to_polygon(box1);
        const auto poly2 = array_to_polygon(box2);

        const float area1 = polygon_area(poly1);
        const float area2 = polygon_area(poly2);

        const std::vector<cv::Point2f> inter_poly = polygon_intersection(poly1, poly2);
        if (inter_poly.empty()) return 0.0f;

        const float inter_area = polygon_area(inter_poly);
        const float union_area = area1 + area2 - inter_area;
        return inter_area / union_area;
    }

    void obb_nms(std::vector<ObbResult>* result, const float iou_threshold, std::vector<int>* index) {
        const size_t N = result->size();
        // Step 1: 根据分数排序得到索引
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::ranges::sort(sorted_indices, [&](const int a, const int b) {
            return (*result)[a].score > (*result)[b].score; // 分数高的排前面
        });

        // Step 2: NMS 主逻辑
        std::vector suppressed(N, false);
        std::vector<int> keep_indices;

        for (size_t m = 0; m < N; ++m) {
            int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep_indices.push_back(i); // 保留当前框
            const auto& box_i = (*result)[i].rotated_box;
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;

                const auto& box_j = (*result)[j].rotated_box;
                const float iou = rotated_iou(box_i, box_j);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }

        // Step 3: 根据 keep_indices 重建结果
        std::vector<ObbResult> new_result;
        new_result.reserve(keep_indices.size());
        for (const auto idx : keep_indices) {
            new_result.push_back(std::move((*result)[idx])); // 移动语义
            if (index) {
                index->push_back(idx);
            }
        }
        result->swap(new_result);
    }
}
