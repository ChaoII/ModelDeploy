//
// Created by aichao on 2025/2/20.
//

#include <vector>
#include "vision/common/result.h"


namespace modeldeploy::vision {
    void ClassifyResult::free() {
        std::vector<int32_t>().swap(label_ids);
        std::vector<float>().swap(scores);
        std::vector<float>().swap(feature);
    }

    void ClassifyResult::clear() {
        label_ids.clear();
        scores.clear();
        feature.clear();
    }

    void ClassifyResult::reserve(const int size) {
        scores.reserve(size);
        label_ids.reserve(size);
    }


    void ClassifyResult::resize(const int size) {
        label_ids.resize(size);
        scores.resize(size);
    }

    void Mask::reserve(const int size) { buffer.reserve(size); }

    void Mask::resize(const int size) { buffer.resize(size); }

    void Mask::free() {
        std::vector<uint8_t>().swap(buffer);
        std::vector<int64_t>().swap(shape);
    }

    void Mask::clear() {
        buffer.clear();
        shape.clear();
    }

    void OCRResult::clear() {
        boxes.clear();
        text.clear();
        rec_scores.clear();
        cls_scores.clear();
        cls_labels.clear();
    }
}
