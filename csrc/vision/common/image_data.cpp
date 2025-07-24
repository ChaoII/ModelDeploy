//
// Created by aichao on 2025/7/18.
//

#include "vision/common/image_data.h"
#include <opencv2/opencv.hpp>

#include "processors/convert_and_permute.h"
#include "vision/utils.h"

namespace modeldeploy {
    class ImageDataImpl {
    public:
        cv::Mat mat;
        ImageDataImpl() = default;

        ImageDataImpl(const int w, const int h, const int c, const int type)
            : mat(h, w, type) {
            // 注意：OpenCV 是 height, width 顺序
        }

        [[nodiscard]] int width() const { return mat.cols; }
        [[nodiscard]] int height() const { return mat.rows; }
        [[nodiscard]] int channels() const { return mat.channels(); }
        [[nodiscard]] int type() const { return mat.type(); }
        [[nodiscard]] size_t size() const { return mat.total() * mat.elemSize(); }

        uint8_t* data() { return mat.data; }
        [[nodiscard]] const uint8_t* data() const { return mat.data; }
    };


    ImageData::ImageData() : impl_(std::make_shared<ImageDataImpl>()) {
    }

    ImageData::ImageData(int width, int height, int channels, int type)
        : impl_(std::make_shared<ImageDataImpl>(width, height, channels, type)) {
    }

    ImageData::ImageData(const ImageData& other) = default;
    ImageData& ImageData::operator=(const ImageData& other) = default;
    ImageData::~ImageData() = default;

    ImageData ImageData::clone() const {
        ImageData copy;
        if (impl_) {
            copy.impl_ = std::make_shared<ImageDataImpl>();
            copy.impl_->mat = impl_->mat.clone(); // 深拷贝数据
        }
        return copy;
    }


    int ImageData::width() const { return impl_->mat.cols; }
    int ImageData::height() const { return impl_->mat.rows; }
    int ImageData::channels() const { return impl_->mat.channels(); }
    int ImageData::type() const { return impl_->mat.type(); }
    size_t ImageData::data_size() const { return impl_->mat.total(); }
    const uint8_t* ImageData::data() const { return impl_->mat.data; }
    uint8_t* ImageData::data() { return impl_->mat.data; }


    void ImageData::to_tensor(Tensor* tensor) const {
        vision::utils::mat_to_tensor(impl_->mat, tensor, false);
    }


    ImageData ImageData::from_mat(const void* mat_ptr) {
        const auto* m = static_cast<const cv::Mat*>(mat_ptr);
        ImageData img;
        img.impl_ = std::make_shared<ImageDataImpl>();
        img.impl_->mat = *m; // header 拷贝，不复制数据
        return img;
    }

    bool ImageData::empty() const {
        return !impl_ || impl_->mat.empty();
    }

    void ImageData::to_mat(void* mat, bool is_copy) const {
        auto* m = static_cast<cv::Mat*>(mat);
        if (!is_copy)
            *m = impl_->mat; // 浅拷贝（共用数据指针）
        else
            *m = impl_->mat.clone();
    }

    void ImageData::images_to_mats(const std::vector<ImageData>& images, const std::vector<void*>& mats) {
        for (size_t i = 0; i < images.size(); ++i) {
            images[i].to_mat(mats[i]);
        }
    }

    // 读取图片
    ImageData ImageData::imread(const std::string& filename) {
        const cv::Mat m = cv::imread(filename);
        if (m.empty()) return {};
        return from_mat(&m);
    }

    // 保存图片
    bool ImageData::imwrite(const std::string& filename) const {
        return cv::imwrite(filename, impl_->mat);
    }

    // 显示图片
    void ImageData::imshow(const std::string& win_name) const {
        cv::imshow(win_name, impl_->mat);
        cv::waitKey(0);
    }


    void ImageData::letter_box(const std::vector<int>& size,
                               const std::vector<float>& padding_value,
                               vision::LetterBoxRecord* letter_box_record) const {
        vision::utils::letter_box(&impl_->mat, size, padding_value, letter_box_record);
    }

    void ImageData::convert_and_permute(const std::vector<float>& alpha, const std::vector<float>& beta,
                                        const bool swap_rb) const {
        vision::ConvertAndPermute::apply(&impl_->mat, alpha, beta, swap_rb);
    }
}
