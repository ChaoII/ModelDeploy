#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::tracking {
    MODELDEPLOY_CXX_EXPORT float iou(const Rect2f& a, const Rect2f& b);
    MODELDEPLOY_CXX_EXPORT std::vector<std::vector<float>> iou_distance(
        const std::vector<Rect2f>& a, const std::vector<Rect2f>& b);
}
