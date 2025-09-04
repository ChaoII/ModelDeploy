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

    ImageData::ImageData(const cv::Mat* mat) {
        impl_ = std::make_shared<ImageDataImpl>();
        impl_->mat = *mat;
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

    ImageData ImageData::from_raw_nocopy(unsigned char* data, const int width, const int height, const int channels) {
        int type;
        if (channels == 3) {
            type = CV_8UC3;
        }
        else if (channels == 4) {
            type = CV_8UC4;
        }
        else if (channels == 1) {
            type = CV_8UC1;
        }
        else {
            throw std::invalid_argument("Invalid number of channels: " + std::to_string(channels));
        }
        ImageData image;
        auto impl = std::make_shared<ImageDataImpl>();
        // 注意：构造 cv::Mat 绑定外部数据（不会复制）
        impl->mat = cv::Mat(height, width, type, data);
        image.impl_ = std::move(impl);
        return image;
    }

    ImageData ImageData::from_raw(const unsigned char* data, const int width, const int height, const int channels) {
        int type;
        if (channels == 3) {
            type = CV_8UC3;
        }
        else if (channels == 4) {
            type = CV_8UC4;
        }
        else if (channels == 1) {
            type = CV_8UC1;
        }
        else {
            throw std::invalid_argument("Invalid number of channels: " + std::to_string(channels));
        }

        ImageData image;
        auto impl = std::make_shared<ImageDataImpl>();
        // 分配内部 cv::Mat 空间
        impl->mat = cv::Mat(height, width, type);

        // 执行深拷贝
        if (data) {
            std::memcpy(impl->mat.data, data, impl->mat.total() * impl->mat.elemSize());
        }
        image.impl_ = std::move(impl);
        return image;
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


    ImageData ImageData::from_mat(const cv::Mat* mat_ptr) {
        ImageData img;
        img.impl_ = std::make_shared<ImageDataImpl>();
        img.impl_->mat = *mat_ptr; // header 拷贝，不复制数据
        return img;
    }

    void ImageData::update_from_mat(const cv::Mat* mat_ptr, const bool is_copy) {
        if (!impl_) {
            impl_ = std::make_shared<ImageDataImpl>();
        }
        if (is_copy) {
            impl_->mat = mat_ptr->clone(); // 深拷贝，数据独立
        }
        else {
            impl_->mat = *mat_ptr; // 浅拷贝，共享数据指针
        }
    }

    bool ImageData::empty() const {
        return !impl_ || impl_->mat.empty();
    }

    void ImageData::to_mat(cv::Mat* mat_ptr, bool is_copy) const {
        if (!is_copy)
            *mat_ptr = impl_->mat; // 浅拷贝（共用数据指针）
        else
            *mat_ptr = impl_->mat.clone();
    }

    void ImageData::images_to_mats(const std::vector<ImageData>& images, const std::vector<cv::Mat*>& mats) {
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
