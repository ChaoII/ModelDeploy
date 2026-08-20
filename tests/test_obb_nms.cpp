//
// Created by the ModelDeploy team on 2026/8/20.
//
#include <vector>
#include <numeric>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "core/md_log.h"
#include "vision/utils.h"
#include "vision/common/struct.h"
#include "vision/common/result.h"

using modeldeploy::vision::RotatedRect;
using modeldeploy::vision::ObbResult;
using modeldeploy::vision::utils::obb_nms;
using modeldeploy::vision::utils::rotated_rect_to_cv_type;

namespace {
    // 与 obb_nms.cpp 中 rotated_iou(cv, cv) 完全相同的 IoU 计算（cv 多边形相交）
    float ref_rotated_iou(const cv::RotatedRect& a, const cv::RotatedRect& b) {
        std::vector<cv::Point2f> inter;
        if (cv::rotatedRectangleIntersection(a, b, inter) <= 0) return 0.0f;
        const float inter_area = static_cast<float>(cv::contourArea(inter));
        const float union_area = a.size.area() + b.size.area() - inter_area;
        return union_area > 0 ? inter_area / union_area : 0.0f;
    }

    // 旧实现（O(N^2) 朴素）作为参考金标准 —— 新实现必须与之逐位一致
    void reference_obb_nms(std::vector<ObbResult>* result, float iou_threshold) {
        const size_t N = result->size();
        std::vector<int> sorted_indices(N);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return (*result)[a].score > (*result)[b].score;
        });
        std::vector<bool> suppressed(N, false);
        std::vector<int> keep;
        for (size_t m = 0; m < N; ++m) {
            const int i = sorted_indices[m];
            if (suppressed[i]) continue;
            keep.push_back(i);
            const auto& b_i = (*result)[i].rotated_box;
            for (size_t n = m + 1; n < N; ++n) {
                const int j = sorted_indices[n];
                if (suppressed[j]) continue;
                const auto& b_j = (*result)[j].rotated_box;
                if (ref_rotated_iou(rotated_rect_to_cv_type(b_i), rotated_rect_to_cv_type(b_j)) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        std::vector<ObbResult> out;
        out.reserve(keep.size());
        for (const int idx : keep) out.push_back((*result)[idx]);
        result->swap(out);
    }

    std::vector<ObbResult> make_case_clustered() {
        return {
            {{10, 10, 20, 20, 0}, 0, 0.9f},   // 高置信，保留
            {{12, 12, 20, 20, 0}, 0, 0.8f},   // 与[0]高IoU → 抑制
            {{60, 60, 20, 20, 0}, 1, 0.7f},   // 保留
            {{62, 62, 20, 20, 0}, 1, 0.6f},   // 与[2]高IoU → 抑制
            {{100, 100, 20, 20, 0}, 2, 0.5f}, // 独立，保留
            {{150, 10, 20, 20, 0}, 0, 0.4f},  // 独立，保留
        };
    }

    std::vector<ObbResult> make_case_dense(int seed) {
        // 确定性伪随机密集场景，AABB 早筛必须零误差
        std::vector<ObbResult> cases;
        unsigned s = static_cast<unsigned>(seed) * 2654435761u + 12345u;
        auto rnd = [&s]() { s = s * 1103515245u + 12345u; return static_cast<float>((s >> 16) & 0x7fff) / 32767.0f; };
        for (int i = 0; i < 400; ++i) {
            const float x = rnd() * 200.0f;
            const float y = rnd() * 200.0f;
            const float w = 10.0f + rnd() * 30.0f;
            const float h = 10.0f + rnd() * 30.0f;
            const float ang = (rnd() - 0.5f) * 180.0f;
            cases.push_back({{x, y, w, h, ang}, i % 5, rnd()});
        }
        return cases;
    }
} // namespace

TEST_CASE("obb_nms AABB-optimized matches reference (exact equivalence)", "[core]") {
    const float thr = 0.5f;
    const std::vector<std::vector<ObbResult>> cases = {
        make_case_clustered(),
        make_case_dense(1),
        make_case_dense(2),
        make_case_dense(7),
    };
    for (auto base : cases) {
        auto ref = base;
        auto opt = base;
        reference_obb_nms(&ref, thr);
        obb_nms(&opt, thr);
        REQUIRE(ref.size() == opt.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(ref[i].score == opt[i].score);
            REQUIRE(ref[i].label_id == opt[i].label_id);
            REQUIRE(ref[i].rotated_box.xc == Catch::Approx(opt[i].rotated_box.xc).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.yc == Catch::Approx(opt[i].rotated_box.yc).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.width == Catch::Approx(opt[i].rotated_box.width).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.height == Catch::Approx(opt[i].rotated_box.height).margin(1e-3f));
            REQUIRE(ref[i].rotated_box.angle == Catch::Approx(opt[i].rotated_box.angle).margin(1e-3f));
        }
    }
}
