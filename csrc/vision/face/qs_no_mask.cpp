//
// Created by AC on 2024-12-25.
//

#include "qs_no_mask.h"

namespace seeta {
    QualityOfNoMask::QualityOfNoMask(std::shared_ptr<seeta::FaceLandmarker> ld_) {
        m_marker = std::move(ld_);
    }

    QualityResult QualityOfNoMask::check(const SeetaImageData& image, const SeetaRect& face,
                                         const SeetaPointF* points, int32_t N) {
        const auto mask_points = m_marker->mark_v2(image, face);
        int mask_count = 0;
        for (auto& point : mask_points) {
            if (point.mask) mask_count++;
        }
        if (mask_count > 0) {
            return {QualityLevel::LOW, 1 - static_cast<float>(mask_count) / static_cast<float>(mask_points.size())};
        }
        return {QualityLevel::HIGH, 1};
    }
}
