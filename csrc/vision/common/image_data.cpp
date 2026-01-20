//
// Created by aichao on 2025/7/18.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/convert.h"
#include "vision/common/image_data.h"
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision {
    ImageData::ImageData(int width, int height, const MdImageType type):
        width_(width), height_(height), type_(type) {
        int ocv_type = md_image_type_to_ocv_type(type);
        mat_ = std::make_unique<cv::Mat>(height, width, ocv_type);
        ownership_ = RawMemoryOwnership::Owned;
    }

    ImageData::~ImageData() = default;

    ImageData::ImageData(const ImageData& other) : type_(other.type_),
                                                   width_(other.width_),
                                                   height_(other.height_),
                                                   channels_(other.channels_) {
        if (other.mat_) {
            mat_ = std::make_unique<cv::Mat>(other.mat_->clone());
            ownership_ = RawMemoryOwnership::Owned;
        }
    }

    ImageData& ImageData::operator=(const ImageData& other) {
        if (this == &other) {
            return *this;
        }
        // 先 clone，保证异常安全
        std::unique_ptr<cv::Mat> new_mat;
        if (other.mat_) {
            new_mat = std::make_unique<cv::Mat>(other.mat_->clone());
        }
        mat_ = std::move(new_mat);
        type_ = other.type_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        ownership_ = RawMemoryOwnership::Owned;
        return *this;
    }

    ImageData::ImageData(ImageData&& other) noexcept
        : mat_(std::move(other.mat_)),
          type_(other.type_),
          width_(other.width_),
          height_(other.height_),
          channels_(other.channels_),
          ownership_(other.ownership_) {
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
    }

    ImageData& ImageData::operator=(ImageData&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        mat_ = std::move(other.mat_);
        type_ = other.type_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        ownership_ = other.ownership_;
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        return *this;
    }

    ImageData::ImageData(const cv::Mat& mat) {
        mat_ = std::make_unique<cv::Mat>(mat.clone());
        type_ = md_image_type_from_ocv_type(mat_->type());
        width_ = mat_->cols;
        height_ = mat_->rows;
        channels_ = mat_->channels();
        ownership_ = RawMemoryOwnership::Owned;
    }

    ImageData::ImageData(cv::Mat&& mat) {
        mat_ = std::make_unique<cv::Mat>(std::move(mat));
        type_ = md_image_type_from_ocv_type(mat_->type());
        width_ = mat_->cols;
        height_ = mat_->rows;
        channels_ = mat_->channels();
        ownership_ = RawMemoryOwnership::Owned;
    }


    ImageData ImageData::clone() const {
        ImageData copy;
        copy.mat_ = std::make_unique<cv::Mat>(mat_->clone());
        copy.type_ = type_;
        copy.width_ = width_;
        copy.height_ = height_;
        copy.channels_ = channels_;
        return copy;
    }

    ImageData ImageData::from_raw(unsigned char* data,
                                  const int width,
                                  const int height,
                                  const MdImageType type,
                                  const bool copy) {
        const int ocv_type = md_image_type_to_ocv_type(type);
        ImageData image;
        cv::Mat tmp_mat;
        if (ocv_type > 0) {
            tmp_mat = cv::Mat(height, width, ocv_type, data);
        }
        else if (type == MdImageType::I420 || type == MdImageType::NV12 || type == MdImageType::NV21) {
            tmp_mat = cv::Mat(height + height / 2, width, CV_8UC1, data);
        }
        else {
            throw std::runtime_error("Invalid MdImageType format" + md_image_type_to_string(type));
        }
        if (copy) {
            image.mat_ = std::make_unique<cv::Mat>(tmp_mat.clone());
            image.ownership_ = RawMemoryOwnership::Owned;
        }
        else {
            image.mat_ = std::make_unique<cv::Mat>(std::move(tmp_mat));
            image.ownership_ = RawMemoryOwnership::Borrowed;
        }
        image.type_ = type;
        image.width_ = width;
        image.height_ = height;
        image.channels_ = image.mat_->channels();
        return image;
    }

    int ImageData::width() const { return width_; }
    int ImageData::height() const { return height_; }
    int ImageData::channels() const { return channels_; }
    MdImageType ImageData::type() const { return type_; }

    size_t ImageData::element_count() const { return mat_->total(); }
    size_t ImageData::element_bytes() const { return mat_->elemSize(); }
    const uint8_t* ImageData::data() const { return mat_->data; }
    uint8_t* ImageData::data() { return mat_->data; }
    bool ImageData::empty() const { return !mat_ || mat_->empty(); }


    ImageData ImageData::crop(const Rect2f& rect) const {
        const cv::Rect2f cv_rect(rect.x, rect.y, rect.width, rect.height);
        auto crop_cv_mat = (*mat_)(cv_rect).clone();
        return ImageData(std::move(crop_cv_mat));
    }

    ImageData ImageData::resize(const int width, const int height) const {
        ImageData image;
        cv::Mat tmp_mat;
        cv::resize(*mat_, tmp_mat, cv::Size(width, height));
        return ImageData(std::move(tmp_mat));
    }

    ImageData ImageData::cast(const std::string& dtype, bool scale) const {
        cv::Mat tmp_mat;
        float scale_factor = 1.0f;
        if (scale) {
            scale_factor = 1.0f / 255.0f;
        }
        if (dtype == "float") {
            if (mat_->type() != CV_32FC(channels_)) {
                mat_->convertTo(tmp_mat, CV_32FC(channels_), scale_factor);
            }
        }
        else if (dtype == "float16") {
            if (mat_->type() != CV_16FC(channels_)) {
                mat_->convertTo(tmp_mat, CV_16FC(channels_), scale_factor);
            }
        }
        else if (dtype == "double" || dtype == "float64") {
            if (mat_->type() != CV_64FC(channels_)) {
                mat_->convertTo(tmp_mat, CV_64FC(channels_), scale_factor);
            }
        }
        else {
            throw std::runtime_error("Cast not support for " + dtype + " now! will skip this operation.");
        }
        return ImageData(std::move(tmp_mat));
    }

    ImageData ImageData::normalize(const std::vector<float>& mean,
                                   const std::vector<float>& std,
                                   const bool swap_rb) const {
        std::vector<cv::Mat> split_im;
        cv::split(*mat_, split_im);
        if (swap_rb) std::swap(split_im[0], split_im[2]);
        for (int c = 0; c < channels_; c++) {
            split_im[c].convertTo(split_im[c], CV_32FC1, mean[c], std[c]);
        }
        cv::Mat tmp_mat;
        cv::merge(split_im, tmp_mat);
        return ImageData(std::move(tmp_mat));
    }

    ImageData ImageData::letter_box(const std::vector<int>& dst_size, const float padding_value) const {
        const auto scale = std::min(dst_size[1] * 1.0 / height(), dst_size[0] * 1.0 / width());
        const int resize_h = static_cast<int>(round(height() * scale));
        const int resize_w = static_cast<int>(round(width() * scale));
        const float pad_w = dst_size[0] - resize_w;
        const float pad_h = dst_size[1] - resize_h;
        const float pad_left = pad_w / 2.0f;
        const float pad_top = pad_h / 2.0f;
        cv::Scalar padding_scalar;
        if (channels() == 1) {
            padding_scalar = cv::Scalar(padding_value);
        }
        else if (channels() == 3) {
            padding_scalar = cv::Scalar(padding_value, padding_value, padding_value);
        }
        else if (channels() == 4) {
            padding_scalar = cv::Scalar(padding_value, padding_value, padding_value, padding_value);
        }
        else {
            throw std::runtime_error("Unsupported image channels.");
        }
        cv::Mat resize_image;
        cv::resize(*mat_, resize_image, cv::Size(resize_w, resize_h));
        cv::Mat tmp_image(dst_size[1], dst_size[0], md_image_type_to_ocv_type(type_), padding_scalar);
        cv::Mat roi = tmp_image(cv::Rect(pad_left, pad_top, resize_w, resize_h));
        cv::resize(*mat_, roi, cv::Size(resize_w, resize_h));
        return ImageData(std::move(tmp_image));
    }

    [[nodiscard]] ImageData ImageData::center_crop(const std::vector<int>& dst_size) const {
        if (width_ < dst_size[0] || height_ < dst_size[1]) {
            throw std::invalid_argument("ImageData::center_crop: dst_size must be smaller than image size.");
        }
        const int offset_x = (width_ - dst_size[0]) / 2;
        const int offset_y = (height_ - dst_size[1]) / 2;
        const Rect2f crop_roi(offset_x, offset_y, dst_size[0], dst_size[1]);
        return crop(crop_roi);
    }

    ImageData ImageData::permute() const {
        return cvt_color(*this, ColorConvertType::CVT_PA_RGB2PL_RGB);
    }

    ImageData ImageData::fuse_normalize_and_permute(const std::vector<float>& mean,
                                                    const std::vector<float>& std) const {
        if (channels() != 3 || channels() != mean.size() || channels() != std.size()) {
            throw std::invalid_argument("channels must be 3 and mean/std size must be 3");
        }
        ImageData dst_image;
        const int single_channel_type = CV_MAKETYPE(mat_->depth(), 1);
        cv::Mat chw_image(channels_, height_ * width_, single_channel_type);
        std::vector<cv::Mat> split_image;
        cv::split(*mat_, split_image);
        std::swap(split_image[0], split_image[2]);
        for (int i = 0; i < split_image.size(); i++) {
            split_image[i].convertTo(split_image[i], CV_32FC1, 1.0 / 255 * mean[i], std[i]);
            split_image[i] = split_image[i].reshape(1, 1);
            split_image[i].copyTo(chw_image.row(i));
        }
        dst_image.mat_ = std::make_unique<cv::Mat>(std::move(chw_image));
        dst_image.type_ = mat_->depth() == CV_8U ? MdImageType::PLA_RGB_U8 : MdImageType::PLA_RGB_F32;
        dst_image.width_ = width();
        dst_image.height_ = height();
        dst_image.channels_ = channels();
        dst_image.ownership_ = RawMemoryOwnership::Owned;
        return dst_image;
    }

    ImageData ImageData::fuse_convert_and_permute() const {
        ImageData dst_image;
        const int single_channel_type = CV_MAKETYPE(mat_->depth(), 1);
        cv::Mat chw_image(channels_, height_ * width_, single_channel_type);
        std::vector<cv::Mat> split_image;
        cv::split(*mat_, split_image);
        std::swap(split_image[0], split_image[2]);
        for (int i = 0; i < split_image.size(); i++) {
            split_image[i].convertTo(split_image[i], CV_32FC1, 1.0 / 255, 0);
            split_image[i] = split_image[i].reshape(1, 1);
            split_image[i].copyTo(chw_image.row(i));
        }
        dst_image.mat_ = std::make_unique<cv::Mat>(std::move(chw_image));
        dst_image.type_ = mat_->depth() == CV_8U ? MdImageType::PLA_RGB_U8 : MdImageType::PLA_RGB_F32;
        dst_image.width_ = width();
        dst_image.height_ = height();
        dst_image.channels_ = channels();
        dst_image.ownership_ = RawMemoryOwnership::Owned;
        return dst_image;
    }


    ImageData ImageData::cvt_color(const ImageData& image, ColorConvertType type) {
        auto ocv_color_convert_type = md_color_convert_type_to_ocv_color_convert_type(type);
        if (ocv_color_convert_type > 0) {
            cv::Mat tmp_mat;
            cv::cvtColor(*image.mat_, tmp_mat, ocv_color_convert_type);
            return ImageData(std::move(tmp_mat));
        }
        if (type == ColorConvertType::CVT_PA_BGR2PL_BGR || type == ColorConvertType::CVT_PA_RGB2PL_RGB) {
            ImageData dst_image;
            const int single_channel_type = CV_MAKETYPE(image.mat_->depth(), 1);
            cv::Mat chw_image(image.channels(), image.height() * image.width(), single_channel_type);
            std::vector<cv::Mat> split_image;
            cv::split(*image.mat_, split_image);
            for (int i = 0; i < split_image.size(); i++) {
                split_image[i] = split_image[i].reshape(1, 1);
                split_image[i].copyTo(chw_image.row(i));
            }
            chw_image = chw_image.reshape(1, {image.channels(), image.height(), image.width()});
            dst_image.mat_ = std::make_unique<cv::Mat>(std::move(chw_image));
            dst_image.type_ = image.mat_->depth() == CV_8U ? MdImageType::PLA_BGR_U8 : MdImageType::PLA_BGR_F32;
            dst_image.width_ = image.width();
            dst_image.height_ = image.height();
            dst_image.channels_ = image.channels();
            dst_image.ownership_ = RawMemoryOwnership::Owned;
            return dst_image;
        }
        if (type == ColorConvertType::CVT_PL_BGR2PA_BGR || type == ColorConvertType::CVT_PL_RGB2PA_RGB) {
            // valid chw format
            if (image.type() != MdImageType::PLA_BGR_U8 && image.type() != MdImageType::PLA_BGR_F32) {
                throw std::runtime_error("Invalid PL_BGR format: expected Planar layout");
            }
            ImageData dst_image;
            // 1 channel per row, total rows equal to channels
            cv::Mat planarImage = image.mat_->reshape(1, image.channels());
            // 2. Split the planar image into separate channel matrices.
            std::vector<cv::Mat> split_images(image.channels());
            for (int i = 0; i < image.channels(); ++i) {
                split_images[i] = planarImage.row(i).reshape(1, image.height()); // reshape each row back to H x W
            }
            // 3. Merge these channel matrices into a single HWC image.
            cv::Mat hwc_image;
            cv::merge(split_images, hwc_image);

            dst_image.mat_ = std::make_unique<cv::Mat>(std::move(hwc_image));
            dst_image.type_ = hwc_image.depth() == CV_8U
                                  ? MdImageType::PKG_BGR_U8
                                  : hwc_image.depth() == CV_32F
                                  ? MdImageType::PKG_BGR_F32
                                  : MdImageType::PKG_BGR_F64;
            dst_image.width_ = image.width();
            dst_image.height_ = image.height();
            dst_image.channels_ = image.channels();
            dst_image.ownership_ = RawMemoryOwnership::Owned;
            return dst_image;
        }
        throw std::runtime_error("Unsupported color conversion type");
    }


    ImageData ImageData::from_mat(const cv::Mat& mat, const bool copy) {
        ImageData img;
        if (copy) {
            img.mat_ = std::make_unique<cv::Mat>(mat.clone());
            img.ownership_ = RawMemoryOwnership::Owned;
        }
        else {
            img.mat_ = std::make_unique<cv::Mat>(mat);
            img.ownership_ = RawMemoryOwnership::Borrowed;
        }
        img.width_ = img.mat_->cols;
        img.height_ = img.mat_->rows;
        img.channels_ = img.mat_->channels();
        img.type_ = md_image_type_from_ocv_type(img.mat_->type());
        return img;
    }

    void ImageData::to_mat(cv::Mat* mat_ptr, const bool copy) const {
        if (!copy) {
            // Shallow copy: share underlying data
            *mat_ptr = *mat_;
        }
        else {
            *mat_ptr = mat_->clone();
        }
    }

    void ImageData::to_tensor(Tensor* tensor, bool copy) {
        const auto dtype = utils::cv_dtype_to_md_dtype(mat_->type());
        const std::vector<int64_t> shape = {channels(), height(), width()};
        if (copy) {
            const size_t num_bytes = element_count() * element_bytes();
            tensor->allocate({channels(), height(), width()}, dtype);
            if (num_bytes != tensor->byte_size()) {
                MD_LOG_ERROR << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = " << tensor->byte_size()
                    << ", size of Mat = " << num_bytes << "." << std::endl;
            }
            memcpy(tensor->data(), data(), num_bytes);
        }
        else {
            // OpenCV Mat 的内存管理由 Mat 自己处理，这里不需要额外操作
            // 注意tensor共享外部内存，所以需要从外部内存中创建tensor，内存由Mat提供，所以deleter可以不给，不需要进行手动释放
            // 确保mat在tensor生命周期结束前有效
            tensor->from_external_memory(data(), shape, dtype);
        }
    }


    ImageData ImageData::imread(const std::string& filename) {
        cv::Mat m = cv::imread(filename);
        if (m.empty()) {
            throw std::runtime_error("Failed to read image: " + filename);
        }
        return ImageData(std::move(m));
    }

    bool ImageData::imwrite(const std::string& filename) const {
        if (!mat_ || mat_->empty()) {
            std::cerr << "ImageData is empty." << std::endl;
            return false;
        }
        return cv::imwrite(filename, *mat_);
    }

    // 显示图片
    void ImageData::imshow(const std::string& win_name) const {
        cv::imshow(win_name, *mat_);
        cv::waitKey(0);
    }
}
