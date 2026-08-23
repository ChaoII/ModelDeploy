#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"
namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT Slice { ImageData tile; Rect2f offset; };
class MODELDEPLOY_CXX_EXPORT InferenceSlicer {
public:
    InferenceSlicer(int tile_w, int tile_h, int overlap_px = 0)
        : tile_w_(tile_w), tile_h_(tile_h), overlap_(overlap_px) {}
    std::vector<Slice> slice(const ImageData& img) const;
private:
    int tile_w_, tile_h_, overlap_;
};
MODELDEPLOY_CXX_EXPORT void reassemble(const std::vector<Slice>& slices,
                                       const std::vector<Detections>& per_slice,
                                       ImageData* out, std::vector<Rect2f>* mapped_boxes);
} // namespace modeldeploy::vision::tool
