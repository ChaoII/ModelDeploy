#pragma once
#include <cstdint>
namespace modeldeploy::vision {
    // 返回设备字形位图指针(懒加载);失败返回 nullptr
    const uint8_t* cjk_device_glyphs();
}
