#pragma once
#include "vision/common/image_data.h"
#include <cstdint>

namespace modeldeploy::video {
// 统一解码帧：内嵌 ImageData（CPU/设备 NV12 等，零拷贝）+ 毫秒时间戳
struct VideoFrame {
    modeldeploy::vision::ImageData image;
    uint64_t pts_ms = 0;
};
} // namespace modeldeploy::video
