#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"

namespace modeldeploy::vision::tracking {
    MODELDEPLOY_CXX_EXPORT std::vector<std::pair<int, int>> linear_sum_assignment(
        const std::vector<std::vector<float>>& cost);
}
