//
// Created by aichao on 2025/2/21.
//

#include "vision/ocr/utils/ocr_utils.h"

namespace modeldeploy::vision::ocr {
    bool compare_box(const std::array<int, 8>& result1,
                     const std::array<int, 8>& result2) {
        if (result1[1] < result2[1]) {
            return true;
        }
        if (result1[1] == result2[1]) {
            return result1[0] < result2[0];
        }
        return false;
    }

    void sort_boxes(std::vector<std::array<int, 8>>* boxes) {
        std::sort(boxes->begin(), boxes->end(), compare_box);
        if (boxes->empty()) {
            return;
        }
        for (int i = 0; i < boxes->size() - 1; i++) {
            for (int j = i; j >= 0; j--) {
                if (std::abs((*boxes)[j + 1][1] - (*boxes)[j][1]) < 10 &&
                    (*boxes)[j + 1][0] < (*boxes)[j][0]) {
                    std::swap((*boxes)[i], (*boxes)[i + 1]);
                }
            }
        }
    }

    std::vector<int> arg_sort(const std::vector<float>& array) {
        const int array_len = static_cast<int>(array.size());
        std::vector array_index(array_len, 0);
        for (int i = 0; i < array_len; ++i)
            array_index[i] = i;
        std::sort(
            array_index.begin(), array_index.end(),
            [&array](const int pos1, const int pos2) { return array[pos1] < array[pos2]; });
        return array_index;
    }
}
