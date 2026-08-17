//
// Created by aichao on 2026/8/17.
//

#include "vision/processors/cpu/draw_nv12.h"
#include "vision/processors/common/font8x16.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::vision {

namespace {
    const int kFontW = 8;
    const int kFontH = 16;

    // BT.601 ???????????????????????
    inline void rgb_to_yuv(uint8_t r, uint8_t g, uint8_t b,
                           uint8_t* py, uint8_t* pu, uint8_t* pv) {
        const int ry = (66 * r + 129 * g + 25 * b + 128) >> 8;
        const int ub = (-38 * r - 74 * g + 112 * b + 128) >> 8;
        const int vr = (112 * r - 94 * g - 18 * b + 128) >> 8;
        *py = static_cast<uint8_t>(std::clamp(ry + 16, 0, 255));
        *pu = static_cast<uint8_t>(std::clamp(ub + 128, 0, 255));
        *pv = static_cast<uint8_t>(std::clamp(vr + 128, 0, 255));
    }

    inline void put_yuv_pixel(uint8_t* y, uint8_t* uv, int w, int h,
                              int step_y, int step_uv, int x, int yc,
                              uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || yc < 0 || x >= w || yc >= h) return;
        uint8_t yy, uu, vv;
        rgb_to_yuv(r, g, b, &yy, &uu, &vv);
        y[static_cast<size_t>(yc) * step_y + x] = yy;
        // UV ???? 2x2 ??? UV ???NV12 ? CbCr ???
        const int ux = x >> 1, uy = yc >> 1;
        uv[static_cast<size_t>(uy) * step_uv + ux * 2] = uu;
        uv[static_cast<size_t>(uy) * step_uv + ux * 2 + 1] = vv;
    }

    // Bresenham ??
    inline void draw_line_cpu(uint8_t* y, uint8_t* uv, int w, int h,
                              int step_y, int step_uv,
                              int x0, int y0, int x1, int y1,
                              uint8_t r, uint8_t g, uint8_t b, int thickness) {
        const int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
        const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        int cx = x0, cy = y0;
        const int half = thickness / 2;
        for (int i = 0; i < 100000; ++i) {
            for (int t = 0; t < thickness; ++t) {
                for (int tt = 0; tt < thickness; ++tt) {
                    put_yuv_pixel(y, uv, w, h, step_y, step_uv,
                                  cx + t - half, cy + tt - half, r, g, b);
                }
            }
            if (cx == x1 && cy == y1) break;
            const int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; cx += sx; }
            if (e2 < dx) { err += dx; cy += sy; }
        }
    }
} // namespace

bool draw_rect_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                        float x, float yo, float rw, float rh,
                        uint8_t r, uint8_t g, uint8_t b, int thickness) {
    if (!y || !uv || w <= 0 || h <= 0) return false;
    if (thickness <= 0) thickness = 2;
    const int x1 = std::max(0, static_cast<int>(std::floor(x)));
    const int y1 = std::max(0, static_cast<int>(std::floor(yo)));
    const int x2 = std::min(w, static_cast<int>(std::ceil(x + rw)));
    const int y2 = std::min(h, static_cast<int>(std::ceil(yo + rh)));
    if (x1 >= x2 || y1 >= y2) return true;
    for (int t = 0; t < thickness; ++t) {
        for (int px = x1; px < x2; ++px) {
            put_yuv_pixel(y, uv, w, h, step_y, step_uv, px, y1 + t, r, g, b);
            put_yuv_pixel(y, uv, w, h, step_y, step_uv, px, y2 - 1 - t, r, g, b);
        }
        for (int py2 = y1 + thickness; py2 < y2 - thickness; ++py2) {
            put_yuv_pixel(y, uv, w, h, step_y, step_uv, x1 + t, py2, r, g, b);
            put_yuv_pixel(y, uv, w, h, step_y, step_uv, x2 - 1 - t, py2, r, g, b);
        }
    }
    return true;
}

bool draw_polygon_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                           const std::vector<Point2f>& pts,
                           uint8_t r, uint8_t g, uint8_t b, int thickness) {
    if (!y || !uv || pts.size() < 2) return false;
    if (thickness <= 0) thickness = 2;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        draw_line_cpu(y, uv, w, h, step_y, step_uv,
                      static_cast<int>(std::lround(pts[i].x)), static_cast<int>(std::lround(pts[i].y)),
                      static_cast<int>(std::lround(pts[i + 1].x)), static_cast<int>(std::lround(pts[i + 1].y)),
                      r, g, b, thickness);
    }
    if (pts.size() > 2) {
        draw_line_cpu(y, uv, w, h, step_y, step_uv,
                      static_cast<int>(std::lround(pts.back().x)), static_cast<int>(std::lround(pts.back().y)),
                      static_cast<int>(std::lround(pts[0].x)), static_cast<int>(std::lround(pts[0].y)),
                      r, g, b, thickness);
    }
    return true;
}

bool draw_points_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                          const std::vector<Point3f>& pts,
                          uint8_t r, uint8_t g, uint8_t b, int radius) {
    if (!y || !uv || pts.empty()) return false;
    if (radius <= 0) radius = 3;
    for (const auto& p : pts) {
        const int cx = static_cast<int>(std::lround(p.x));
        const int cy = static_cast<int>(std::lround(p.y));
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= radius * radius) {
                    put_yuv_pixel(y, uv, w, h, step_y, step_uv, cx + dx, cy + dy, r, g, b);
                }
            }
        }
    }
    return true;
}

// ASCII 8x16 ?????font_size ?????1=?? 8x16?
bool draw_text_nv12_cpu(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                        float x, float yo, const std::string& text,
                        uint8_t r, uint8_t g, uint8_t b, int font_size) {
    if (!y || !uv || text.empty()) return false;
    if (font_size <= 0) font_size = 1;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(yo);
    for (size_t ci = 0; ci < text.size(); ++ci) {
        const unsigned char ch = static_cast<unsigned char>(text[ci]);
        if (ch < 0x20 || ch > 0x7E) continue;
        for (int fy = 0; fy < kFontH; ++fy) {
            const uint8_t bits = font8x16::kFont8x16[ch - 0x20][fy];
            for (int fx = 0; fx < kFontW; ++fx) {
                if (!(bits & (0x80 >> fx))) continue;
                const int sx = x0 + static_cast<int>(ci) * kFontW * font_size + fx * font_size;
                const int sy = y0 + fy * font_size;
                for (int syy = 0; syy < font_size; ++syy) {
                    for (int sxx = 0; sxx < font_size; ++sxx) {
                        put_yuv_pixel(y, uv, w, h, step_y, step_uv, sx + sxx, sy + syy, r, g, b);
                    }
                }
            }
        }
    }
    return true;
}
} // namespace modeldeploy::vision

