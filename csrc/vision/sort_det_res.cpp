//
// Created by aichao on 2025/3/25.
//
#include "csrc/vision/utils.h"
#include "csrc/vision/common/result.h"

namespace modeldeploy::vision::utils
{
    void merge(DetectionResult* result, const size_t low, const size_t mid, const size_t high) {
        std::vector<cv::Rect2f>& boxes = result->boxes;
        std::vector<float>& scores = result->scores;
        std::vector<int32_t>& label_ids = result->label_ids;
        const std::vector temp_boxes(boxes);
        const std::vector temp_scores(scores);
        const std::vector temp_label_ids(label_ids);
        size_t i = low;
        size_t j = mid + 1;
        size_t k = i;
        for (; i <= mid && j <= high; k++) {
            if (temp_scores[i] >= temp_scores[j]) {
                scores[k] = temp_scores[i];
                label_ids[k] = temp_label_ids[i];
                boxes[k] = temp_boxes[i];
                i++;
            }
            else {
                scores[k] = temp_scores[j];
                label_ids[k] = temp_label_ids[j];
                boxes[k] = temp_boxes[j];
                j++;
            }
        }
        while (i <= mid) {
            scores[k] = temp_scores[i];
            label_ids[k] = temp_label_ids[i];
            boxes[k] = temp_boxes[i];
            k++;
            i++;
        }
        while (j <= high) {
            scores[k] = temp_scores[j];
            label_ids[k] = temp_label_ids[j];
            boxes[k] = temp_boxes[j];
            k++;
            j++;
        }
    }

    void merge_sort(DetectionResult* result, const size_t low, const size_t high) {
        if (low < high) {
            const size_t mid = (high - low) / 2 + low;
            merge_sort(result, low, mid);
            merge_sort(result, mid + 1, high);
            merge(result, low, mid, high);
        }
    }

    void sort_detection_result(DetectionResult* result) {
        constexpr size_t low = 0;
        size_t high = result->scores.size();
        if (high == 0) {
            return;
        }
        high = high - 1;
        merge_sort(result, low, high);
    }

    template <typename T>
    bool lex_sort_by_xy_compare(const cv::Rect2f& box_a,
                                const cv::Rect2f& box_b) {
        // WARN: The status shoule be false if (a==b).
        // https://blog.csdn.net/xxxwrq/article/details/83080640
        auto is_equal = [](const T& a, const T& b) -> bool {
            return std::abs(a - b) < 1e-6f;
        };
        const T& x0_a = box_a.x;
        const T& y0_a = box_a.y;
        const T& x0_b = box_b.x;
        const T& y0_b = box_b.y;
        if (is_equal(x0_a, x0_b)) {
            return !is_equal(y0_a, y0_b) && y0_a > y0_b;
        }
        return x0_a > x0_b;
    }

    // Only for int dtype
    // template <>
    // bool lex_sort_by_xy_compare(const Rect& box_a,
    //                             const Rect& box_b) {
    //     const int& x0_a = box_a.x;
    //     const int& y0_a = box_a.y;
    //     const int& x0_b = box_b.x;
    //     const int& y0_b = box_b.y;
    //     if (x0_a == x0_b) {
    //         return y0_a == y0_b ? false : y0_a > y0_b;
    //     }
    //     return x0_a > x0_b;
    // }

    void reorder_detection_result_by_indices(DetectionResult* result,
                                             const std::vector<size_t>& indices) {
        // reorder boxes, scores, label_ids, masks
        DetectionResult backup = *result;
        const bool contain_masks = backup.contain_masks;
        const int boxes_num = static_cast<int>(backup.boxes.size());
        result->clear();
        result->resize(boxes_num);
        // boxes, scores, labels_ids
        for (int i = 0; i < boxes_num; ++i) {
            result->boxes[i] = backup.boxes[indices[i]];
            result->scores[i] = backup.scores[indices[i]];
            result->label_ids[i] = backup.label_ids[indices[i]];
        }
        if (contain_masks) {
            result->contain_masks = true;
            for (int i = 0; i < boxes_num; ++i) {
                const auto& shape = backup.masks[indices[i]].shape;
                const int mask_num_el = static_cast<int>(shape[0] * shape[1]);
                result->masks[i].shape = shape;
                result->masks[i].resize(mask_num_el);
                std::memcpy(result->masks[i].data(), backup.masks[indices[i]].data(),
                            mask_num_el * sizeof(uint8_t));
            }
        }
    }

    void LexltectionResultByXY(DetectionResult* result) {
        if (result->boxes.empty()) {
            return;
        }
        std::vector<size_t> indices;
        indices.resize(result->boxes.size());
        for (size_t i = 0; i < result->boxes.size(); ++i) {
            indices[i] = i;
        }
        // lex sort by x(w) then y(h)
        auto& boxes = result->boxes;
        std::ranges::sort(indices, [&boxes](const size_t a, const size_t b) {
            return lex_sort_by_xy_compare<int>(boxes[a], boxes[b]);
        });
        reorder_detection_result_by_indices(result, indices);
    }

    void LexSortOCRDetResultByXY(std::vector<std::array<int, 8>>* result) {
        if (result->empty()) {
            return;
        }
        std::vector<size_t> indices;
        indices.resize(result->size());
        std::vector<cv::Rect2f> boxes;
        boxes.resize(result->size());
        for (size_t i = 0; i < result->size(); ++i) {
            indices[i] = i;
            // 4 points to 2 points for LexSort
            boxes[i] = {
                (float)(*result)[i][0], (float)(*result)[i][1],
                (float)((*result)[i][6] - (*result)[i][0]),
                (float)((*result)[i][7] - (*result)[i][1])
            };
        }
        // lex sort by x(w) then y(h)
        std::ranges::sort(indices, [&boxes](const size_t a, const size_t b) {
            return lex_sort_by_xy_compare<int>(boxes[a], boxes[b]);
        });
        // reorder boxes
        const std::vector<std::array<int, 8>> backup = *result;
        const int boxes_num = static_cast<int>(backup.size());
        result->clear();
        result->resize(boxes_num);
        // boxes
        for (int i = 0; i < boxes_num; ++i) {
            (*result)[i] = backup[indices[i]];
        }
    }
}
