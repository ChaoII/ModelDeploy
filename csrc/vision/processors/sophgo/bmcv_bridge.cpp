//
// BMCV 预处理桥接实现（见 bmcv_bridge.h）。
// 仅在 ENABLE_SOPHGO 编译时链接真实 BMCV；否则退化返回 -1（调用方走 CPU 兜底）。
//

#include "bmcv_bridge.h"
#include <cstdio>
#include <cstring>
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
} // namespace modeldeploy::vision
