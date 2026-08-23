#include "vision/solutions/object_cropper.h"
namespace modeldeploy::vision::solution {
void ObjectCropper::crop(const ImageData& img, const Rect2f& box, ImageData* out) const {
    if (!out) return;
    *out = img.crop(box);
}
} // namespace modeldeploy::vision::solution
