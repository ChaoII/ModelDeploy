//
// Created by aichao on 2026/8/17.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "vision/common/basic_types.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision {
    // 在 NV12 Y/UV 平面就地绘制。坐标裁剪到帧内。r/g/b 为 BGR（0-255），内部转 YUV BT.601。
    bool draw_rect_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                            float x, float yo, float rw, float rh,
                            uint8_t r, uint8_t g, uint8_t b, int thickness);
    bool draw_polygon_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                               const std::vector<Point2f>& pts,
                               uint8_t r, uint8_t g, uint8_t b, int thickness);
    bool draw_points_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                              const std::vector<Point3f>& pts,
                              uint8_t r, uint8_t g, uint8_t b, int radius);
    bool draw_text_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                            float x, float yo, const std::string& text,
                            uint8_t r, uint8_t g, uint8_t b, int font_size);
}
