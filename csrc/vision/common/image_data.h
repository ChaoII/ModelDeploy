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
        // NV12/NV21：Y/UV 平面指针与步长；非 NV12 返回 nullptr/0
        [[nodiscard]] const uint8_t* y() const;
        [[nodiscard]] const uint8_t* uv() const;
        [[nodiscard]] int step_y() const;
        [[nodiscard]] int step_uv() const;
        // 通用平面数量（NV12=2，packed=1）
        [[nodiscard]] size_t plane_count() const;
        // 设备侧构造：绑定外部设备内存平面（零拷贝，不拥有内存）
        static ImageData from_device_planes(uint8_t* y, uint8_t* uv, int w, int h,
                                            int step_y, int step_uv, Device device);

        [[nodiscard]] bool is_shared_with(const ImageData& other) const;

        [[nodiscard]] ImageData clone() const;
        static ImageData cvt_color(const ImageData& image, ColorConvertType type);
        // Caller must guarantee data lifetime >= ImageData lifetime
        static ImageData from_raw(unsigned char* data, int width, int height, MdImageType type, bool copy = false);
        static void images_to_tensor(const std::vector<ImageData>& images, Tensor* tensor);
        void to_mat(cv::Mat& mat, bool copy = false) const;
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
