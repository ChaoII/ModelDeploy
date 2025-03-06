//
// Created by aichao on 2025/2/21.
//

#include "ocr_utils.h"

namespace modeldeploy::vision::ocr {
    std::vector<int> xyxyxyxy2xyxy(const std::array<int, 8>& box) {
        int x_collect[4] = {box[0], box[2], box[4], box[6]};
        int y_collect[4] = {box[1], box[3], box[5], box[7]};
        const int left = *std::min_element(x_collect, x_collect + 4);
        const int right = *std::max_element(x_collect, x_collect + 4);
        const int top = *std::min_element(y_collect, y_collect + 4);
        const int bottom = *std::max_element(y_collect, y_collect + 4);
        std::vector<int> box1(4, 0);
        box1[0] = left;
        box1[1] = top;
        box1[2] = right;
        box1[3] = bottom;
        return box1;
    }

    float dis(const std::vector<int>& box1, const std::vector<int>& box2) {
        const float x1_1 = static_cast<float>(box1[0]);
        const float y1_1 = static_cast<float>(box1[1]);
        const float x2_1 = static_cast<float>(box1[2]);
        const float y2_1 = static_cast<float>(box1[3]);

        const float x1_2 = static_cast<float>(box2[0]);
        const float y1_2 = static_cast<float>(box2[1]);
        const float x2_2 = static_cast<float>(box2[2]);
        const float y2_2 = static_cast<float>(box2[3]);

        const float dis = std::abs(x1_2 - x1_1) + std::abs(y1_2 - y1_1) +
            std::abs(x2_2 - x2_1) + std::abs(y2_2 - y2_1);
        const float dis_2 = std::abs(x1_2 - x1_1) + std::abs(y1_2 - y1_1);
        const float dis_3 = std::abs(x2_2 - x2_1) + std::abs(y2_2 - y2_1);
        return dis + std::min(dis_2, dis_3);
    }

    float iou(const std::vector<int>& box1, const std::vector<int>& box2) {
        const int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
        const int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

        // computing the sum_area
        const int sum_area = area1 + area2;

        // find the each point of intersect rectangle
        const int x1 = std::max(box1[0], box2[0]);
        const int y1 = std::max(box1[1], box2[1]);
        const int x2 = std::min(box1[2], box2[2]);
        const int y2 = std::min(box1[3], box2[3]);

        // judge if there is an intersect
        if (y1 >= y2 || x1 >= x2) {
            return 0.0;
        }
        const int intersect = (x2 - x1) * (y2 - y1);
        return intersect / (sum_area - intersect + 0.00000001);
    }

    bool comparison_dis(const std::vector<float>& dis1,
                        const std::vector<float>& dis2) {
        if (dis1[1] < dis2[1]) {
            return true;
        }
        if (dis1[1] == dis2[1]) {
            return dis1[0] < dis2[0];
        }
        return false;
    }
}
