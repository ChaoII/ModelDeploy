#include "vision/face/face_det/scrfd_preproc.h"
#include <vision/utils.h>

namespace modeldeploy::vision {
    static constexpr float scrfd_inv128 = 1.0f / 128.0f;
    static constexpr float scrfd_offset = -127.5f / 128.0f;

    static void scrfd_preprocess_bgr_cpu_inner(
        const uint8_t* src, const int src_w, const int src_h,
        float* dst, const int dst_w, const int dst_h,
        const float scale, const float pad_w, const float pad_h,
        const float pad_value) {
        const int plane_size = dst_w * dst_h;
        const float pad_x_end = pad_w + static_cast<float>(src_w) * scale;
        const float pad_y_end = pad_h + static_cast<float>(src_h) * scale;
        const float pad_f = pad_value * scrfd_inv128 + scrfd_offset;

        for (int dy = 0; dy < dst_h; ++dy) {
            for (int dx = 0; dx < dst_w; ++dx) {
                const int out_idx = dy * dst_w + dx;
                if (dx < pad_w || dx >= pad_x_end || dy < pad_h || dy >= pad_y_end) {
                    dst[0 * plane_size + out_idx] = pad_f;
                    dst[1 * plane_size + out_idx] = pad_f;
                    dst[2 * plane_size + out_idx] = pad_f;
                    continue;
                }
                const int sx = static_cast<int>((dx - pad_w) / scale);
                const int sy = static_cast<int>((dy - pad_h) / scale);
                const int src_idx = (sy * src_w + sx) * 3;
                const float b = src[src_idx + 0];
                const float g = src[src_idx + 1];
                const float r = src[src_idx + 2];
                dst[0 * plane_size + out_idx] = r * scrfd_inv128 + scrfd_offset;
                dst[1 * plane_size + out_idx] = g * scrfd_inv128 + scrfd_offset;
                dst[2 * plane_size + out_idx] = b * scrfd_inv128 + scrfd_offset;
            }
        }
    }

    static void scrfd_preprocess_nv12_cpu_inner(
        const uint8_t* src_y_data, const uint8_t* src_uv_data,
        const int src_w, const int src_h,
        const int step_y, const int step_uv,
        float* dst, const int dst_w, const int dst_h,
        const float scale, const float pad_w, const float pad_h,
        const float pad_value) {
        const int plane_size = dst_w * dst_h;
        const float pad_f = pad_value * scrfd_inv128 + scrfd_offset;

        for (int dy = 0; dy < dst_h; dy++) {
            for (int dx = 0; dx < dst_w; dx++) {
                const float src_xf = (dx - pad_w) / scale;
                const float src_yf = (dy - pad_h) / scale;
                const int sx = static_cast<int>(src_xf);
                const int sy = static_cast<int>(src_yf);
                const int dst_idx = dy * dst_w + dx;

                if (sx < 0 || sx >= src_w || sy < 0 || sy >= src_h) {
                    dst[0 * plane_size + dst_idx] = pad_f;
                    dst[1 * plane_size + dst_idx] = pad_f;
                    dst[2 * plane_size + dst_idx] = pad_f;
                } else {
                    const float lum = src_y_data[sy * step_y + sx];
                    const int uv_x = sx >> 1;
                    const int uv_y = sy >> 1;
                    const int safe_uv_x = std::min(uv_x, (src_w >> 1) - 1);
                    const int safe_uv_y = std::min(uv_y, (src_h >> 1) - 1);
                    const uint8_t* uv_row = src_uv_data + safe_uv_y * step_uv;
                    float u_val = uv_row[safe_uv_x * 2 + 0];
                    float v_val = uv_row[safe_uv_x * 2 + 1];
                    u_val -= 128.0f;
                    v_val -= 128.0f;
                    float r = lum + 1.402f * v_val;
                    float g = lum - 0.344136f * u_val - 0.714136f * v_val;
                    float b = lum + 1.772f * u_val;
                    r = fminf(fmaxf(r, 0.0f), 255.0f);
                    g = fminf(fmaxf(g, 0.0f), 255.0f);
                    b = fminf(fmaxf(b, 0.0f), 255.0f);
                    dst[0 * plane_size + dst_idx] = r * scrfd_inv128 + scrfd_offset;
                    dst[1 * plane_size + dst_idx] = g * scrfd_inv128 + scrfd_offset;
                    dst[2 * plane_size + dst_idx] = b * scrfd_inv128 + scrfd_offset;
                }
            }
        }
    }

    bool scrfd_preprocess_cpu(const ImageData& image, Tensor* output,
                              const std::vector<int>& dst_size, float pad_val,
                              LetterBoxRecord* letter_box_record) {
        return scrfd_preprocess_bgr_cpu(image.data(), {image.width(), image.height()},
                                        output, dst_size, pad_val, letter_box_record);
    }

    bool scrfd_preprocess_bgr_cpu(const uint8_t* src, const std::vector<int>& src_size,
                                  Tensor* output, const std::vector<int>& dst_size,
                                  float pad_val, LetterBoxRecord* letter_box_record) {
        if (!src) return false;
        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        scrfd_preprocess_bgr_cpu_inner(src, src_w, src_h, output->data_ptr<float>(),
                                       dst_w, dst_h,
                                       letter_box_record->scale, letter_box_record->pad_w,
                                       letter_box_record->pad_h, pad_val);
        output->expand_dim(0);
        return true;
    }

    bool scrfd_preprocess_nv12_cpu(const uint8_t* src_y, const uint8_t* src_uv,
                                   const std::vector<int>& src_size,
                                   int step_y, int step_uv,
                                   Tensor* output, const std::vector<int>& dst_size,
                                   float pad_value, LetterBoxRecord* letter_box_record) {
        if (!src_y || !src_uv) return false;
        const int src_w = src_size[0];
        const int src_h = src_size[1];
        const int dst_w = dst_size[0];
        const int dst_h = dst_size[1];
        output->allocate({3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        *letter_box_record = utils::cal_letter_box_param({src_w, src_h}, {dst_w, dst_h});
        scrfd_preprocess_nv12_cpu_inner(src_y, src_uv, src_w, src_h,
                                        step_y, step_uv,
                                        output->data_ptr<float>(), dst_w, dst_h,
                                        letter_box_record->scale, letter_box_record->pad_w,
                                        letter_box_record->pad_h, pad_value);
        output->expand_dim(0);
        return true;
    }
}