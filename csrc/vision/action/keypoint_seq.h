#pragma once

#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::action {

// 一段视频片段的骨骼序列：frames[t][v] 为第 t 帧第 v 个关节坐标（Point3f；进行 2D 动作时 z=0）。
struct MODELDEPLOY_CXX_EXPORT KeyPointSeq {
    std::vector<std::vector<Point3f>> frames;
};

} // namespace modeldeploy::vision::action
