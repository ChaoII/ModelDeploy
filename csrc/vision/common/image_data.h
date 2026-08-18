//
// Created by aichao on 2025/7/18.
//

#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include "core/tensor.h"
#include "vision/common/struct.h"
#include "vision/common/basic_types.h"


namespace cv {
    class Mat;
}

namespace modeldeploy::vision {
    class ImageDataImpl;

    class MODELDEPLOY_CXX_EXPORT ImageData {
    public:
        ImageData() = default;
        ImageData(int width, int height, MdImageType type);
        explicit ImageData(const cv::Mat& mat);
        explicit ImageData(cv::Mat&& mat);

        // 全部是浅拷贝
        ImageData(const ImageData& other) = default;
        ImageData& operator=(const ImageData& other) = default;
        ImageData(ImageData&& other) noexcept = default;
        ImageData& operator=(ImageData&& other) noexcept = default;

        ~ImageData() = default;

        // 平面描述：packed 格式 1 个平面（data），NV12/NV21 2 个平面（y, uv），I420 3 个平面（y, u, v）
        struct Plane {
            const uint8_t* data = nullptr;
            int step = 0;
        };

        [[nodiscard]] int width() const;
        [[nodiscard]] int height() const;
        [[nodiscard]] int channels() const;
        [[nodiscard]] MdImageType type() const;
        [[nodiscard]] size_t element_count() const;
        [[nodiscard]] size_t element_bytes() const;
        [[nodiscard]] size_t bytes() const;
        [[nodiscard]] const uint8_t* data() const;
        [[nodiscard]] uint8_t* data();
        [[nodiscard]] bool empty() const;

        [[nodiscard]] Device device() const;
        // 通用平面数量（NV12=2，packed=1）
        [[nodiscard]] size_t plane_count() const;
        // 统一取第 i 个平面（多平面/设备）；越界返回空 Plane
        [[nodiscard]] Plane plane(size_t i) const;
        // == type() 的别名
        [[nodiscard]] MdImageType format() const;
        // 设备侧构造：绑定外部设备内存平面（零拷贝，不拥有内存）
        static ImageData from_device_planes(uint8_t* y, uint8_t* uv, int w, int h,
                                            int step_y, int step_uv, Device device);
        // CPU 单平面（PKG_BGR_U8）借用构造
        static ImageData from_bgr24(const uint8_t* bgr, int w, int h);
        // 设备/平面 → CPU 深拷贝；已是 CPU 则浅 clone
        [[nodiscard]] bool toCpu(ImageData* out) const;
        // 仅 device()==CPU 的 mat 有效（借用）；否则 false
        [[nodiscard]] bool asMat(cv::Mat* out) const;
        // thread_local 错误通道（每操作起始清空，失败写入；成功返回 nullptr）
        static const char* last_error();

        [[nodiscard]] bool is_shared_with(const ImageData& other) const;

        [[nodiscard]] ImageData clone() const;
        static ImageData cvt_color(const ImageData& image, ColorConvertType type);
        // Caller must guarantee data lifetime >= ImageData lifetime
        static ImageData from_raw(unsigned char* data, int width, int height, MdImageType type, bool copy = false);
        static void images_to_tensor(const std::vector<ImageData>& images, Tensor* tensor);
        void to_tensor(Tensor* tensor, bool copy = false);
        static std::vector<uint8_t> imencode(const ImageData& image, const std::string& ext);
        static ImageData imdecode(const std::vector<uint8_t>& buf);
        static ImageData imread(const std::string& filename);
        [[nodiscard]] bool imwrite(const std::string& filename) const;
        void imshow(const std::string& win_name) const;


        // 预处理相关
        ImageData& rotate(RotateFlags flag);
        [[nodiscard]] ImageData crop(const Rect2f& rect) const;
        [[nodiscard]] ImageData rotate_crop(std::array<float, 8> box) const;
        [[nodiscard]] ImageData resize(int width, int height) const;

    private:
        std::shared_ptr<ImageDataImpl> impl_;
    };
}
