//
// BMCV 预处理桥接实现（见 bmcv_bridge.h）。
// 仅在 ENABLE_SOPHGO 编译时链接真实 BMCV；否则退化返回 -1（调用方走 CPU 兜底）。
//

#include "bmcv_bridge.h"
#include <cstdio>

#ifdef ENABLE_SOPHGO
#include "bmlib_runtime.h"
#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#endif

namespace modeldeploy::vision {

#ifdef ENABLE_SOPHGO
    int md_bmcv_letterbox_normalize(void* handle,
                                    const uint8_t* bgr, int src_w, int src_h,
                                    float* dst, int dst_w, int dst_h,
                                    int pad_w, int pad_h, int resize_w, int resize_h,
                                    float alpha0, float alpha1, float alpha2,
                                    unsigned char pad_val, int swap_rb) {
        if (!handle || !bgr || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        const bm_image_format_ext src_fmt = swap_rb ? FORMAT_BGR_PACKED : FORMAT_RGB_PACKED;
        const bm_image_data_format_ext u8 = DATA_TYPE_EXT_1N_BYTE;

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

            st = bm_image_create(h, dst_h, dst_w, src_fmt, u8, &letter_img, nullptr);
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

            const bm_image_format_ext rgb_planar = FORMAT_RGB_PLANAR;
            const bm_image_data_format_ext f32 = DATA_TYPE_EXT_FLOAT32;
            st = bm_image_create(h, dst_h, dst_w, rgb_planar, f32, &out_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(out_img, 0);
            if (st != BM_SUCCESS) break;
            bmcv_convert_to_attr ct = {alpha0, 0.0f, alpha1, 0.0f, alpha2, 0.0f};
            st = bmcv_image_convert_to(h, 1, ct, &letter_img, &out_img);
            if (st != BM_SUCCESS) break;

            const size_t plane = static_cast<size_t>(dst_h) * dst_w;
            void* out_host[3] = {dst, dst + plane, dst + 2 * plane};
            st = bm_image_copy_device_to_host(out_img, out_host);
        } while (false);

        if (st == BM_SUCCESS) {
            bm_image_destroy(src_img);
            bm_image_destroy(letter_img);
            bm_image_destroy(out_img);
            return 0;
        }
        bm_image_destroy(src_img);
        bm_image_destroy(letter_img);
        bm_image_destroy(out_img);
        return -1;
    }

    int md_bmcv_letterbox_normalize_device(void* handle,
                                           const uint8_t* bgr, int src_w, int src_h,
                                           void* out_img_ptr, int dst_w, int dst_h,
                                           int pad_w, int pad_h, int resize_w, int resize_h,
                                           float alpha0, float alpha1, float alpha2,
                                           unsigned char pad_val, int swap_rb) {
        if (!handle || !bgr || !out_img_ptr || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        bm_image* out_img = static_cast<bm_image*>(out_img_ptr);  // 调用方持有：FP32 RGB_PLANAR 已创建+alloc
        const bm_image_format_ext src_fmt = swap_rb ? FORMAT_BGR_PACKED : FORMAT_RGB_PACKED;
        const bm_image_data_format_ext u8 = DATA_TYPE_EXT_1N_BYTE;

        bm_image src_img{};
        bm_image letter_img{};
        bm_status_t st = BM_SUCCESS;

        do {
            st = bm_image_create(h, src_h, src_w, src_fmt, u8, &src_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(src_img, 0);
            if (st != BM_SUCCESS) break;
            void* src_host[] = {const_cast<uint8_t*>(bgr)};
            st = bm_image_copy_host_to_device(src_img, src_host);
            if (st != BM_SUCCESS) break;

            st = bm_image_create(h, dst_h, dst_w, src_fmt, u8, &letter_img, nullptr);
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

            bmcv_convert_to_attr ct = {alpha0, 0.0f, alpha1, 0.0f, alpha2, 0.0f};
            st = bmcv_image_convert_to(h, 1, ct, &letter_img, out_img);
        } while (false);

        bm_image_destroy(src_img);
        bm_image_destroy(letter_img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_letterbox_normalize_device_full(void* handle,
                                                const uint8_t* bgr, int src_w, int src_h,
                                                void** out_img, void* dev_mem,
                                                int dst_w, int dst_h,
                                                int pad_w, int pad_h, int resize_w, int resize_h,
                                                float alpha0, float alpha1, float alpha2,
                                                unsigned char pad_val, int swap_rb) {
        if (!handle || !out_img || !dev_mem || !bgr || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        const bm_image_format_ext rgb_planar = FORMAT_RGB_PLANAR;
        const bm_image_data_format_ext f32 = DATA_TYPE_EXT_FLOAT32;
        bm_image* oi = new bm_image{};
        fprintf(stderr, "[bridge] create out img\n"); fflush(stderr);
        if (bm_image_create(h, dst_h, dst_w, rgb_planar, f32, oi, nullptr) != BM_SUCCESS) {
            delete oi;
            return -1;
        }
        fprintf(stderr, "[bridge] alloc out img\n"); fflush(stderr);
        if (bm_image_alloc_dev_mem(*oi, 0) != BM_SUCCESS) {
            bm_image_destroy(*oi);
            delete oi;
            return -1;
        }
        fprintf(stderr, "[bridge] normalize_device\n"); fflush(stderr);
        if (md_bmcv_letterbox_normalize_device(handle, bgr, src_w, src_h, oi, dst_w, dst_h,
                                               pad_w, pad_h, resize_w, resize_h,
                                               alpha0, alpha1, alpha2, pad_val, swap_rb) != 0) {
            bm_image_destroy(*oi);
            delete oi;
            return -1;
        }
        fprintf(stderr, "[bridge] get_device_mem\n"); fflush(stderr);
        if (bm_image_get_device_mem(*oi, static_cast<bm_device_mem_t*>(dev_mem)) != BM_SUCCESS) {
            bm_image_destroy(*oi);
            delete oi;
            return -1;
        }
        fprintf(stderr, "[bridge] ok\n"); fflush(stderr);
        *out_img = oi;
        return 0;
    }

    void* md_bmcv_image_create(void* handle, int dst_w, int dst_h) {
        if (!handle || dst_w <= 0 || dst_h <= 0) return nullptr;
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        bm_image* oi = new bm_image{};
        if (bm_image_create(h, dst_h, dst_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32,
                            oi, nullptr) != BM_SUCCESS) {
            delete oi;
            return nullptr;
        }
        return oi;
    }

    int md_bmcv_letterbox_normalize_attach(void* handle,
                                           const uint8_t* bgr, int src_w, int src_h,
                                           void* out_img, void* input_mem,
                                           int dst_w, int dst_h,
                                           int pad_w, int pad_h, int resize_w, int resize_h,
                                           float alpha0, float alpha1, float alpha2,
                                           unsigned char pad_val, int swap_rb) {
        if (!handle || !bgr || !out_img || !input_mem ||
            src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            return -1;
        }
        bm_handle_t h = static_cast<bm_handle_t>(handle);
        bm_image* oi = static_cast<bm_image*>(out_img);
        // out_img 必须 attach 到 bmrt_tensor 分配的输入设备内存（官方零拷贝关键）
        if (bm_image_attach(*oi, static_cast<bm_device_mem_t*>(input_mem)) != BM_SUCCESS) {
            return -1;
        }
        const bm_image_format_ext src_fmt = swap_rb ? FORMAT_BGR_PACKED : FORMAT_RGB_PACKED;
        const bm_image_data_format_ext u8 = DATA_TYPE_EXT_1N_BYTE;
        const int aligned_w = (dst_w + 63) / 64 * 64;
        int letter_strides[3] = {aligned_w, aligned_w, aligned_w};

        bm_image src_img{};
        bm_image letter_img{};
        bm_status_t st = BM_SUCCESS;

        do {
            st = bm_image_create(h, src_h, src_w, src_fmt, u8, &src_img, nullptr);
            if (st != BM_SUCCESS) break;
            st = bm_image_alloc_dev_mem(src_img, 0);
            if (st != BM_SUCCESS) break;
            void* src_host[] = {const_cast<uint8_t*>(bgr)};
            st = bm_image_copy_host_to_device(src_img, src_host);
            if (st != BM_SUCCESS) break;

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

            // convert_to 直接写入 out_img（其设备内存 == input_mem）
            bmcv_convert_to_attr ct = {alpha0, 0.0f, alpha1, 0.0f, alpha2, 0.0f};
            st = bmcv_image_convert_to(h, 1, ct, &letter_img, oi);
        } while (false);

        bm_image_destroy(src_img);
        bm_image_destroy(letter_img);
        return st == BM_SUCCESS ? 0 : -1;
    }

    int md_bmcv_image_destroy(void* img) {
        if (!img) return 0;
        bm_image* oi = static_cast<bm_image*>(img);
        bm_image_destroy(*oi);
        delete oi;
        return 0;
    }
#else
    int md_bmcv_letterbox_normalize(void* handle,
                                    const uint8_t* bgr, int src_w, int src_h,
                                    float* dst, int dst_w, int dst_h,
                                    int pad_w, int pad_h, int resize_w, int resize_h,
                                    float alpha0, float alpha1, float alpha2,
                                    unsigned char pad_val, int swap_rb) {
        (void)handle; (void)bgr; (void)src_w; (void)src_h;
        (void)dst; (void)dst_w; (void)dst_h;
        (void)pad_w; (void)pad_h; (void)resize_w; (void)resize_h;
        (void)alpha0; (void)alpha1; (void)alpha2;
        (void)pad_val; (void)swap_rb;
        return -1;
    }

    int md_bmcv_letterbox_normalize_device_full(void* handle,
                                                const uint8_t* bgr, int src_w, int src_h,
                                                void** out_img, void* dev_mem,
                                                int dst_w, int dst_h,
                                                int pad_w, int pad_h, int resize_w, int resize_h,
                                                float alpha0, float alpha1, float alpha2,
                                                unsigned char pad_val, int swap_rb) {
        (void)handle; (void)bgr; (void)src_w; (void)src_h;
        (void)out_img; (void)dev_mem; (void)dst_w; (void)dst_h;
        (void)pad_w; (void)pad_h; (void)resize_w; (void)resize_h;
        (void)alpha0; (void)alpha1; (void)alpha2;
        (void)pad_val; (void)swap_rb;
        return -1;
    }

    int md_bmcv_image_destroy(void* img) {
        (void)img;
        return 0;
    }

    void* md_bmcv_image_create(void* handle, int dst_w, int dst_h) {
        (void)handle; (void)dst_w; (void)dst_h;
        return nullptr;
    }

    int md_bmcv_letterbox_normalize_attach(void* handle,
                                           const uint8_t* bgr, int src_w, int src_h,
                                           void* out_img, void* input_mem,
                                           int dst_w, int dst_h,
                                           int pad_w, int pad_h, int resize_w, int resize_h,
                                           float alpha0, float alpha1, float alpha2,
                                           unsigned char pad_val, int swap_rb) {
        (void)handle; (void)bgr; (void)src_w; (void)src_h;
        (void)out_img; (void)input_mem; (void)dst_w; (void)dst_h;
        (void)pad_w; (void)pad_h; (void)resize_w; (void)resize_h;
        (void)alpha0; (void)alpha1; (void)alpha2;
        (void)pad_val; (void)swap_rb;
        return -1;
    }
#endif

} // namespace modeldeploy::vision
