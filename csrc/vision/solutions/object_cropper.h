#pragma once
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ObjectCropper {
public:
    void crop(const ImageData& img, const Rect2f& box, ImageData* out) const;
};
} // namespace modeldeploy::vision::solution
