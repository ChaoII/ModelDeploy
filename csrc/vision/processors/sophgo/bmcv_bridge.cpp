//
// BMCV 预处理桥接实现（见 bmcv_bridge.h）。
// 仅在 ENABLE_SOPHGO 编译时链接真实 BMCV；否则退化返回 -1（调用方走 CPU 兜底）。
//

#include "bmcv_bridge.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#include "bmlib_runtime.h"
#include "bmcv_api.h"
#include "bmcv_api_ext.h"

namespace modeldeploy::vision {
    int md_bmcv_letterbox_normalize_to_devmem(void* handle,
                                              const uint8_t* bgr, int src_w, int src_h,
                                              void* dev_mem, int dst_w, int dst_h,
                                              int pad_w, int pad_h, int resize_w, int resize_h,
                                              float alpha0, float alpha1, float alpha2,
                                              unsigned char pad_val, int swap_rb) {
        if (!handle || !bgr || !dev_mem || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        const bm_image_format_ext src_fmt = swap_rb ? FORMAT_BGR_PACKED : FORMAT_RGB_PACKED;
        const bm_image_data_format_ext u8 = DATA_TYPE_EXT_1N_BYTE;
        const bm_image_format_ext rgb_planar = FORMAT_RGB_PLANAR;
        const bm_image_data_format_ext f32 = DATA_TYPE_EXT_FLOAT32;

        bm_image src_img{};
        bm_image letter_img{};
        bm_image out_img{};
        bm_status_t st = BM_SUCCESS;

        do {
            st = bm_image_create(h, src_h, src_w, src_fmt, u8, &src_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(src_img, 0);
            if (st != BM_SUCCESS) break;
            void* src_host[] = {const_cast<uint8_t*>(bgr)};
            st = bm_image_copy_host_to_device(src_img, src_host);
            if (st != BM_SUCCESS) break;

            const int aligned_w = (dst_w + 63) / 64 * 64;
            int letter_strides[3] = {aligned_w, aligned_w, aligned_w};
            st = bm_image_create(h, dst_h, dst_w, FORMAT_RGB_PLANAR, u8, &letter_img, letter_strides);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(letter_img, 0);
            if (st != BM_SUCCESS) break;
            bmcv_padding_attr_t pad_attr = {
                static_cast<unsigned>(pad_w), static_cast<unsigned>(pad_h),
                static_cast<unsigned>(resize_w), static_cast<unsigned>(resize_h),
                pad_val, pad_val, pad_val, 1
            };
            bmcv_rect_t crop = {0, 0, static_cast<unsigned>(src_w), static_cast<unsigned>(src_h)};
            st = bmcv_image_vpp_convert_padding(h, 1, src_img, &letter_img, &pad_attr, &crop);
            if (st != BM_SUCCESS) break;

            // out_img attach 到输入设备内存（调用方持有），convert_to 直接写入（零拷贝）
            st = bm_image_create(h, dst_h, dst_w, rgb_planar, f32, &out_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_attach(out_img, static_cast<bm_device_mem_t*>(dev_mem));
            if (st != BM_SUCCESS) break;
            bmcv_convert_to_attr ct = {alpha0, 0.0f, alpha1, 0.0f, alpha2, 0.0f};
            st = bmcv_image_convert_to(h, 1, ct, &letter_img, &out_img);
        }
        while (false);

        bm_image_destroy(src_img);
        bm_image_destroy(letter_img);
        bm_image_destroy(out_img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_nv12_letterbox_normalize_to_devmem(void* handle,
                                                   const uint8_t* src_y, const uint8_t* src_uv,
                                                   int src_w, int src_h,
                                                   int step_y, int step_uv,
                                                   void* dev_mem, int dst_w, int dst_h,
                                                   float alpha0, float alpha1, float alpha2,
                                                   unsigned char pad_val,
                                                   bool src_is_device) {
        if (!handle || !src_y || !src_uv || !dev_mem ||
            src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        const bm_image_format_ext rgb_planar = FORMAT_RGB_PLANAR;
        const bm_image_data_format_ext f32 = DATA_TYPE_EXT_FLOAT32;

        // 若 Y/UV 平面行步长与尺寸不一致（带 stride），整理为连续平面后再上传。
        // 设备源时不做整理（要求 Y/UV 已连续）。
        const bool contiguous = (step_y <= 0 || step_y == src_w) &&
            (step_uv <= 0 || step_uv == src_w);
        std::vector<uint8_t> y_buf, uv_buf;
        const uint8_t* y_data = src_y;
        const uint8_t* uv_data = src_uv;
        if (!contiguous && !src_is_device) {
            const int sy = step_y > 0 ? step_y : src_w;
            const int suv = step_uv > 0 ? step_uv : src_w;
            y_buf.resize(static_cast<size_t>(src_h) * src_w);
            for (int r = 0; r < src_h; ++r) {
                std::memcpy(y_buf.data() + static_cast<size_t>(r) * src_w,
                            src_y + static_cast<size_t>(r) * sy, src_w);
            }
            const int uv_h = src_h / 2;
            uv_buf.resize(static_cast<size_t>(uv_h) * src_w);
            for (int r = 0; r < uv_h; ++r) {
                std::memcpy(uv_buf.data() + static_cast<size_t>(r) * src_w,
                            src_uv + static_cast<size_t>(r) * suv, src_w);
            }
            y_data = y_buf.data();
            uv_data = uv_buf.data();
        }

        bm_image nv12_img{};
        bm_image letter_img{};
        bm_image out_img{};
        bm_status_t st = BM_SUCCESS;

        do {
            st = bm_image_create(h, src_h, src_w, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE,
                                 &nv12_img, nullptr);
            if (st != BM_SUCCESS) break;
            if (src_is_device) {
                // 设备源：src_y/src_uv 是 TPU 设备地址（Y 后紧跟 UV 的连续 NV12），
                // 直接 attach 到 src 内存，跳过 alloc + H2D。
                const size_t y_bytes = static_cast<size_t>(src_h) * src_w;
                const size_t uv_bytes = static_cast<size_t>(src_h / 2) * src_w;
                bm_device_mem_t src_mem{};
                bm_mem_set_device_addr(&src_mem, reinterpret_cast<unsigned long long>(src_y));
                bm_mem_set_device_size(&src_mem, static_cast<unsigned int>(y_bytes + uv_bytes));
                st = bm_image_attach(nv12_img, &src_mem);
                if (st != BM_SUCCESS) break;
            } else {
                st = bm_image_alloc_dev_mem(nv12_img, 0);
                if (st != BM_SUCCESS) break;
                void* host_planes[] = {const_cast<uint8_t*>(y_data), const_cast<uint8_t*>(uv_data)};
                st = bm_image_copy_host_to_device(nv12_img, host_planes);
                if (st != BM_SUCCESS) break;
            }

            const int aligned_w = (dst_w + 63) / 64 * 64;
            int letter_strides[3] = {aligned_w, aligned_w, aligned_w};
            st = bm_image_create(h, dst_h, dst_w, FORMAT_RGB_PLANAR,
                                 DATA_TYPE_EXT_1N_BYTE, &letter_img, letter_strides);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(letter_img, 0);
            if (st != BM_SUCCESS) break;
            const float scale = (static_cast<float>(dst_w) / src_w <
                                    static_cast<float>(dst_h) / src_h)
                                    ? static_cast<float>(dst_w) / src_w
                                    : static_cast<float>(dst_h) / src_h;
            int rw = static_cast<int>(src_w * scale);
            int rh = static_cast<int>(src_h * scale);
            if (rw > dst_w) rw = dst_w;
            if (rh > dst_h) rh = dst_h;
            const int pw = (dst_w - rw) / 2;
            const int ph = (dst_h - rh) / 2;
            bmcv_padding_attr_t pad_attr = {
                static_cast<unsigned>(pw), static_cast<unsigned>(ph),
                static_cast<unsigned>(rw), static_cast<unsigned>(rh),
                pad_val, pad_val, pad_val, 1
            };
            bmcv_rect_t crop = {0, 0, static_cast<unsigned>(src_w), static_cast<unsigned>(src_h)};
            st = bmcv_image_vpp_convert_padding(h, 1, nv12_img, &letter_img, &pad_attr, &crop);
            if (st != BM_SUCCESS) break;

            st = bm_image_create(h, dst_h, dst_w, rgb_planar, f32, &out_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_attach(out_img, static_cast<bm_device_mem_t*>(dev_mem));
            if (st != BM_SUCCESS) break;
            bmcv_convert_to_attr ct = {alpha0, 0.0f, alpha1, 0.0f, alpha2, 0.0f};
            st = bmcv_image_convert_to(h, 1, ct, &letter_img, &out_img);
        }
        while (false);

        bm_image_destroy(nv12_img);
        bm_image_destroy(letter_img);
        bm_image_destroy(out_img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    // ── NV12 设备侧就地绘制实现 ──
    // 把 TPU 设备显存 Y/UV 地址 attach 成 NV12 bm_image（两平面），就地绘制后 detach。
    // 颜色语义与 CPU/CUDA 后端一致：r/g/b 为 BGR 分量（BMCV 后台转 YUV）。

    namespace {
        // 用给定的 Y/UV 设备地址构造 NV12 bm_image 并 attach（不拥有显存）。
        // 成功返回 BM_SUCCESS 并填充 img；失败返回错误码。
        bm_status_t attach_nv12_image(bm_handle_t h, void* y_mem, void* uv_mem,
                                      int w, int h_image, bm_image* img) {
            if (!h || !y_mem || !uv_mem || w <= 0 || h_image <= 0) return BM_ERR_FAILURE;
            *img = bm_image{};
            bm_status_t st = bm_image_create(h, h_image, w, FORMAT_NV12,
                                             DATA_TYPE_EXT_1N_BYTE, img, nullptr);
            if (st != BM_SUCCESS) return st;
            bm_device_mem_t planes[2]{};
            bm_mem_set_device_addr(&planes[0], reinterpret_cast<unsigned long long>(y_mem));
            bm_mem_set_device_size(&planes[0], static_cast<unsigned int>(h_image * w));
            bm_mem_set_device_addr(&planes[1], reinterpret_cast<unsigned long long>(uv_mem));
            bm_mem_set_device_size(&planes[1], static_cast<unsigned int>(h_image / 2 * w));
            st = bm_image_attach(*img, planes);
            if (st != BM_SUCCESS) bm_image_destroy(img);
            return st;
        }

        // 分配连续 NV12 输出设备内存（Y + UV 紧邻）并 attach 到 img（FORMAT_NV12, w x h_img）。
        // 成功返回 BM_SUCCESS 并写出设备基址 base（Y 平面）与保活 owner；失败时内部清理并返回错误码。
        bm_status_t alloc_attach_nv12_out(bm_handle_t h, int w, int h_img, bm_image* img,
                                          void** base, std::shared_ptr<void>* owner) {
            const size_t y_sz = static_cast<size_t>(h_img) * w;
            const size_t uv_sz = y_sz / 2;
            const size_t tot = y_sz + uv_sz;
            bm_device_mem_t dm{};
            bm_status_t st = bm_malloc_device_byte(h, &dm, static_cast<unsigned int>(tot));
            if (st != BM_SUCCESS) return st;
            const unsigned long long addr = bm_mem_get_device_addr(dm);
            bm_device_mem_t planes[2]{};
            bm_mem_set_device_addr(&planes[0], addr);
            bm_mem_set_device_size(&planes[0], static_cast<unsigned int>(y_sz));
            bm_mem_set_device_addr(&planes[1], addr + y_sz);
            bm_mem_set_device_size(&planes[1], static_cast<unsigned int>(uv_sz));
            st = bm_image_create(h, h_img, w, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE, img, nullptr);
            if (st != BM_SUCCESS) { bm_free_device(h, dm); return st; }
            st = bm_image_attach(*img, planes);
            if (st != BM_SUCCESS) { bm_image_destroy(img); bm_free_device(h, dm); return st; }
            *base = reinterpret_cast<void*>(addr);
            // owner 保活并释放设备显存；平面内存由本 owner 统一管理。
            *owner = std::shared_ptr<void>(*base,
                                           [h, dm](void*) mutable { bm_free_device(h, dm); });
            return BM_SUCCESS;
        }
    } // namespace

    int md_bmcv_draw_rect_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                               int x1, int y1, int x2, int y2,
                               int r, int g, int b, int thickness) {
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image img{};
        bm_status_t st = attach_nv12_image(hd, y_mem, uv_mem, w, h, &img);
        if (st != BM_SUCCESS) return -1;
        if (thickness <= 0) thickness = 1;
        // 坐标裁剪到帧内（与 CPU/CUDA 后端一致），避免越界设备写
        const int xa = std::max(x1, 0);
        const int ya = std::max(y1, 0);
        const int xb = std::min(x2, w);
        const int yb = std::min(y2, h);
        bmcv_rect_t rect{};
        rect.start_x = static_cast<unsigned int>(xa);
        rect.start_y = static_cast<unsigned int>(ya);
        rect.crop_w = static_cast<unsigned int>(std::max(xb - xa, 0));
        rect.crop_h = static_cast<unsigned int>(std::max(yb - ya, 0));
        if (rect.crop_w == 0 || rect.crop_h == 0) { st = BM_ERR_FAILURE; }
        else {
            st = bmcv_image_draw_rectangle(hd, img, 1, &rect, thickness,
                                           static_cast<unsigned char>(r),
                                           static_cast<unsigned char>(g),
                                           static_cast<unsigned char>(b));
        }
        bm_image_detach(img);
        bm_image_destroy(&img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_draw_polygon_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                  const float* xs, const float* ys, int npts,
                                  int r, int g, int b, int thickness) {
        if (!xs || !ys || npts < 3) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image img{};
        bm_status_t st = attach_nv12_image(hd, y_mem, uv_mem, w, h, &img);
        if (st != BM_SUCCESS) return -1;
        if (thickness <= 0) thickness = 1;
        std::vector<bmcv_point_t> start(npts), end(npts);
        for (int i = 0; i < npts; ++i) {
            const int j = (i + 1) % npts;
            start[i] = {static_cast<int>(xs[i]), static_cast<int>(ys[i])};
            end[i] = {static_cast<int>(xs[j]), static_cast<int>(ys[j])};
        }
        bmcv_color_t color{static_cast<unsigned char>(r),
                           static_cast<unsigned char>(g),
                           static_cast<unsigned char>(b)};
        st = bmcv_image_draw_lines(hd, img, start.data(), end.data(), npts, color, thickness);
        bm_image_detach(img);
        bm_image_destroy(&img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_draw_points_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                 const float* xs, const float* ys, int npts, int radius,
                                 int r, int g, int b) {
        if (!xs || !ys || npts < 1) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image img{};
        bm_status_t st = attach_nv12_image(hd, y_mem, uv_mem, w, h, &img);
        if (st != BM_SUCCESS) return -1;
        if (radius <= 0) radius = 1;
        // 每个点绘制一个边长 = 2*radius 的小方块外圈。注意：不用 bmcv_image_fill_rectangle——
        // 其在 BM1688(vpss 路径)对两平面 attach 的 NV12 帧会报地址错误/卡死，故用
        // bmcv_image_draw_rectangle 画空心方框近似“点”（与“按能力近似”一致）。
        std::vector<bmcv_rect_t> rects(npts);
        for (int i = 0; i < npts; ++i) {
            const int cx = static_cast<int>(xs[i]);
            const int cy = static_cast<int>(ys[i]);
            const int x0 = std::max(cx - radius, 0);
            const int y0 = std::max(cy - radius, 0);
            const int x1 = std::min(cx + radius, w - 1);
            const int y1 = std::min(cy + radius, h - 1);
            rects[i].start_x = static_cast<unsigned int>(x0);
            rects[i].start_y = static_cast<unsigned int>(y0);
            rects[i].crop_w = static_cast<unsigned int>(std::max(x1 - x0, 0));
            rects[i].crop_h = static_cast<unsigned int>(std::max(y1 - y0, 0));
        }
        const int thickness = std::max(1, radius >= 3 ? 2 : 1);
        st = bmcv_image_draw_rectangle(hd, img, npts, rects.data(), thickness,
                                       static_cast<unsigned char>(r),
                                       static_cast<unsigned char>(g),
                                       static_cast<unsigned char>(b));
        bm_image_detach(img);
        bm_image_destroy(&img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_draw_text_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                               int x, int y, const char* text,
                               int r, int g, int b, int font_size) {
        if (!text || std::strlen(text) == 0) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image img{};
        bm_status_t st = attach_nv12_image(hd, y_mem, uv_mem, w, h, &img);
        if (st != BM_SUCCESS) return -1;
        if (font_size <= 0) font_size = 1;
        bmcv_point_t org{std::max(x, 0), std::max(y, 0)};
        bmcv_color_t color{static_cast<unsigned char>(r),
                           static_cast<unsigned char>(g),
                           static_cast<unsigned char>(b)};
        // font_size 语义对齐 CPU/CUDA（近似映射：scale = font_size*2，thickness=2）
        const float font_scale = static_cast<float>(font_size) * 2.0f;
        st = bmcv_image_put_text(hd, img, text, org, color, font_scale, 2);
        bm_image_detach(img);
        bm_image_destroy(&img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_draw_lines_nv12(void* handle, void* y_mem, void* uv_mem, int w, int h,
                                const float* sx, const float* sy,
                                const float* ex, const float* ey, int ns,
                                int r, int g, int b, int thickness) {
        if (!sx || !sy || !ex || !ey || ns < 1) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image img{};
        bm_status_t st = attach_nv12_image(hd, y_mem, uv_mem, w, h, &img);
        if (st != BM_SUCCESS) return -1;
        if (thickness <= 0) thickness = 1;
        std::vector<bmcv_point_t> start(ns), end(ns);
        for (int i = 0; i < ns; ++i) {
            start[i] = {static_cast<int>(sx[i]), static_cast<int>(sy[i])};
            end[i] = {static_cast<int>(ex[i]), static_cast<int>(ey[i])};
        }
        bmcv_color_t color{static_cast<unsigned char>(r),
                           static_cast<unsigned char>(g),
                           static_cast<unsigned char>(b)};
        st = bmcv_image_draw_lines(hd, img, start.data(), end.data(), ns, color, thickness);
        bm_image_detach(img);
        bm_image_destroy(&img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    // ── TPU 设备帧中间算子（显存→显存，输出新分配设备内存，零拷贝）──
    // 完整 BMCV 调用以 libsophon 真实 API 为准实现；本机无 Sophgo 环境无法编译/验证，
    // 需在 Sophgo 设备 + ENABLE_SOPHGO 下由 tests/test_sophgo_device_ops.cpp 集成验收。

    int md_bmcv_crop_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                 int src_w, int src_h,
                                 float x, float y, float w, float h,
                                 void** out_y, void** out_uv, int* out_w, int* out_h,
                                 std::shared_ptr<void>* owner) {
        if (!handle || !y_mem || !uv_mem || !out_y || !out_uv || !out_w || !out_h || !owner ||
            src_w <= 0 || src_h <= 0 || w <= 0 || h <= 0) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
        int x1 = static_cast<int>(std::ceil(x + w)), y1 = static_cast<int>(std::ceil(y + h));
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(src_w, x1); y1 = std::min(src_h, y1);
        x0 &= ~1; y0 &= ~1; x1 &= ~1; y1 &= ~1;   // NV12 UV 采样偶数对齐
        const int cw = x1 - x0, ch = y1 - y0;
        if (cw <= 0 || ch <= 0 || (cw & 1) || (ch & 1)) return -1;

        bm_image in{}, out{};
        bm_status_t st = BM_SUCCESS;
        void* base = nullptr;
        do {
            st = attach_nv12_image(hd, y_mem, uv_mem, src_w, src_h, &in);
            if (st != BM_SUCCESS) break;
            st = alloc_attach_nv12_out(hd, cw, ch, &out, &base, owner);
            if (st != BM_SUCCESS) break;
            bmcv_rect_t crop = {static_cast<unsigned>(x0), static_cast<unsigned>(y0),
                                static_cast<unsigned>(cw), static_cast<unsigned>(ch)};
            bmcv_padding_attr_t pad = {0, 0, static_cast<unsigned>(cw), static_cast<unsigned>(ch),
                                       0, 0, 0, 1};
            st = bmcv_image_vpp_convert_padding(hd, 1, &in, &out, &pad, &crop);
        } while (false);
        bm_image_destroy(&in);
        bm_image_destroy(&out);
        if (st != BM_SUCCESS) {
            owner->reset();   // 释放已分配设备显存，避免泄漏
            return -1;
        }
        *out_y = base;
        *out_uv = reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(base) +
                                          static_cast<size_t>(ch) * cw);
        *out_w = cw;
        *out_h = ch;
        return 0;
    }

    int md_bmcv_rotate_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                   int src_w, int src_h, int flag,
                                   void** out_y, void** out_uv, int* out_w, int* out_h,
                                   std::shared_ptr<void>* owner) {
        if (!handle || !y_mem || !uv_mem || !out_y || !out_uv || !out_w || !out_h || !owner ||
            src_w <= 0 || src_h <= 0 || flag < 0 || flag > 2) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        const bool swap = (flag != 1);   // 90/270 交换 W/H，180 不变
        const int o_w = swap ? src_h : src_w;
        const int o_h = swap ? src_w : src_h;
        bmcv_rotate_t angle;
        switch (flag) {
        case 0: angle = BMCV_ROTATE_90; break;
        case 1: angle = BMCV_ROTATE_180; break;
        default: angle = BMCV_ROTATE_270; break;
        }
        bm_image in{}, out{};
        bm_status_t st = BM_SUCCESS;
        void* base = nullptr;
        do {
            st = attach_nv12_image(hd, y_mem, uv_mem, src_w, src_h, &in);
            if (st != BM_SUCCESS) break;
            st = alloc_attach_nv12_out(hd, o_w, o_h, &out, &base, owner);
            if (st != BM_SUCCESS) break;
            st = bmcv_image_rotate(hd, in, &out, angle);
        } while (false);
        bm_image_destroy(&in);
        bm_image_destroy(&out);
        if (st != BM_SUCCESS) {
            owner->reset();
            return -1;
        }
        *out_y = base;
        *out_uv = reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(base) +
                                          static_cast<size_t>(o_h) * o_w);
        *out_w = o_w;
        *out_h = o_h;
        return 0;
    }

    int md_bmcv_cvtcolor_nv12_devmem(void* handle, void* y_mem, void* uv_mem,
                                     int src_w, int src_h, int cvt_kind,
                                     void** out_data, int* out_w, int* out_h,
                                     std::shared_ptr<void>* owner) {
        if (!handle || !y_mem || !uv_mem || !out_data || !out_w || !out_h || !owner ||
            src_w <= 0 || src_h <= 0 || cvt_kind < 0 || cvt_kind > 2) return -1;
        bm_handle_t hd = static_cast<bm_handle_t>(handle);
        bm_image_format_ext out_fmt;
        switch (cvt_kind) {
        case 0: out_fmt = FORMAT_BGR_PACKED; break;
        case 1: out_fmt = FORMAT_BGR_PLANAR; break;
        default: out_fmt = FORMAT_GRAY; break;
        }
        const size_t out_bytes = (cvt_kind == 2)
                                     ? static_cast<size_t>(src_w) * src_h
                                     : static_cast<size_t>(src_w) * src_h * 3;
        bm_image in{}, out{};
        bm_status_t st = BM_SUCCESS;
        void* base = nullptr;
        do {
            st = attach_nv12_image(hd, y_mem, uv_mem, src_w, src_h, &in);
            if (st != BM_SUCCESS) break;
            bm_device_mem_t dm{};
            st = bm_malloc_device_byte(hd, &dm, static_cast<unsigned int>(out_bytes));
            if (st != BM_SUCCESS) break;
            const unsigned long long addr = bm_mem_get_device_addr(dm);
            st = bm_image_create(hd, src_h, src_w, out_fmt, DATA_TYPE_EXT_1N_BYTE, &out, nullptr);
            if (st != BM_SUCCESS) { bm_free_device(hd, dm); break; }
            st = bm_image_attach(out, &dm);
            if (st != BM_SUCCESS) { bm_image_destroy(&out); bm_free_device(hd, dm); break; }
            base = reinterpret_cast<void*>(addr);
            bmcv_convert_to_attr ct = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
            st = bmcv_image_convert_to(hd, 1, ct, &in, &out);
            *owner = std::shared_ptr<void>(base,
                                           [hd, dm](void*) mutable { bm_free_device(hd, dm); });
        } while (false);
        bm_image_destroy(&in);
        bm_image_destroy(&out);
        if (st != BM_SUCCESS) {
            owner->reset();
            return -1;
        }
        *out_data = base;
        *out_w = src_w;
        *out_h = src_h;
        return 0;
    }
} // namespace modeldeploy::vision
