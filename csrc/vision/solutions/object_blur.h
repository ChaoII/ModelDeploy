#pragma once
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ObjectBlur {
public:
    explicit ObjectBlur(int ksize = 15) : ksize_(ksize) {}
    void blur(const ImageData& img, const Rect2f& box, ImageData* out) const;
private:
    int ksize_;
};
} // namespace modeldeploy::vision::solution
