//
// insightface buffalo_l det_10g 后处理。
// 独立 SCRFD 解码（det_10g 输出 2D shape，与标准 Scrfd 3D shape 不同）：
// stride 8/16/32 双 anchor + distance2bbox/kps + NMS。
//
#pragma once

#include <vector>
#include "core/tensor.h"
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/face/insightface/insightface_types.h"

namespace modeldeploy::vision::face {

    class MODELDEPLOY_CXX_EXPORT InsightFaceDetPostprocessor {
    public:
        bool run(const std::vector<Tensor>& infer_results,
                 const std::vector<LetterBoxRecord>& letter_box_records,
                 std::vector<std::vector<InsightFaceBox>>* results);

        float nms_thresh_ = 0.4f;
    };

} // namespace modeldeploy::vision::face
