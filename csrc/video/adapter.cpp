#include "csrc/video/adapter.h"

namespace modeldeploy::video {
using modeldeploy::vision::ImageData;
ImageData make_image_from_planes_view(const IPlaneView& v) {
    if (!v.y || !v.uv || v.width <= 0 || v.height <= 0) return {};
    ImageData::Plane pl[2] = { {v.y, v.y_step}, {v.uv, v.uv_step} };
    return ImageData::from_planes(pl, 2, MdImageType::NV12,
                                  v.width, v.height, v.device, v.owner);
}
} // namespace modeldeploy::video
