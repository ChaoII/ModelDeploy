//
// Created by aichao on 2025/7/18.
//

#pragma once

#include <memory>
#include <vector>
#include <cstdint>
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
        ImageData();
        ImageData(int width, int height, MdImageType type);
        explicit ImageData(const cv::Mat& mat);
        explicit ImageData(cv::Mat&& mat);

        ImageData(const ImageData& other);
        ImageData& operator=(const ImageData& other);

        ImageData(ImageData&& other) noexcept;
        ImageData& operator=(ImageData&& other) noexcept;

        ~ImageData();

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

        [[nodiscard]] ImageData clone() const;
        static ImageData cvt_color(const ImageData& image, ColorConvertType type);
        // Caller must guarantee data lifetime >= ImageData lifetime
        static ImageData from_raw(unsigned char* data, int width, int height, MdImageType type, bool copy = false);
        static void images_to_tensor(std::vector<ImageData> images, Tensor* tensor);
        void to_mat(cv::Mat& mat, bool copy = false) const;
        void to_tensor(Tensor* tensor, bool copy = false);
        static ImageData imread(const std::string& filename);
        [[nodiscard]] bool imwrite(const std::string& filename) const;
        void imshow(const std::string& win_name) const;

        // 预处理相关
        ImageData& rotate(RotateFlags flag);
        [[nodiscard]] ImageData crop(const Rect2f& rect) const;
        [[nodiscard]] ImageData rotate_crop(std::array<float, 8> box) const;
        [[nodiscard]] ImageData resize(int width, int height) const;
        [[nodiscard]] ImageData cast(const std::string& dtype = "float", bool scale = true) const;
        [[nodiscard]] ImageData pad(int top, int bottom, int left, int right, float value) const;
        [[nodiscard]] ImageData convert(const std::vector<float>& alpha = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f},
                                        const std::vector<float>& beta = {0.0f, 0.0f, 0.0f}) const;
        [[nodiscard]] ImageData normalize(const std::vector<float>& mean, const std::vector<float>& std,
                                          bool scale = true, bool swap_rb = true) const;
        [[nodiscard]] ImageData letter_box(const std::vector<int>& dst_size, float padding_value) const;
        [[nodiscard]] ImageData center_crop(const std::vector<int>& dst_size) const;
        [[nodiscard]] ImageData permute() const;
        [[nodiscard]] ImageData fuse_normalize_and_permute(const std::vector<float>& mean,
                                                           const std::vector<float>& std, bool scale = true) const;
        [[nodiscard]] ImageData fuse_convert_and_permute(
            const std::vector<float>& alpha = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f},
            const std::vector<float>& beta = {0.0f, 0.0f, 0.0f}) const;

    private:
        std::unique_ptr<ImageDataImpl> impl_;
    };
}
