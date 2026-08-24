#pragma once
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include <memory>
#include <cstdint>

namespace modeldeploy::video {
// 后端解码对象的统一平面视图（零拷贝中间表示）
struct IPlaneView {
    const uint8_t* y = nullptr; int y_step = 0;
    const uint8_t* uv = nullptr; int uv_step = 0;
    int width = 0; int height = 0;
    modeldeploy::Device device = modeldeploy::Device::CPU;
    std::shared_ptr<void> owner;   // 保活解码 buffer
};
// 由平面视图零拷贝构造 ImageData（NV12 双平面）
MODELDEPLOY_CXX_EXPORT modeldeploy::vision::ImageData make_image_from_planes_view(const IPlaneView& v);
} // namespace modeldeploy::video
