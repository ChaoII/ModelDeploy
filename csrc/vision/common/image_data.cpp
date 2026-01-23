//
// Created by aichao on 2025/7/18.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/convert.h"
#include "vision/common/image_data.h"
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision {
    class ImageDataImpl {
    public:
        cv::Mat mat;
        int width = 0;
        int height = 0;
        int channels = 0;
        MdImageType type = MdImageType::PKG_BGR_U8;
        RawMemoryOwnership ownership = RawMemoryOwnership::Owned;

        ImageDataImpl() = default;

        ImageDataImpl(int w, int h, MdImageType t)
            : width(w), height(h), type(t) {
            const int ocv_type = md_image_type_to_ocv_type(t);
            mat = cv::Mat(height, width, ocv_type);
            channels = mat.channels();
        }

        ImageDataImpl(const cv::Mat& m, bool copy = true) {
            if (copy) {
                mat = m.clone();
            }
            else {
                mat = m;
            }
            width = mat.cols;
            height = mat.rows;
            channels = mat.channels();
            type = md_image_type_from_ocv_type(mat.type());
        }

        ImageDataImpl(cv::Mat&& m) : mat(std::move(m)) {
            width = mat.cols;
            height = mat.rows;
            channels = mat.channels();
            type = md_image_type_from_ocv_type(mat.type());
        }

        ImageDataImpl clone() const {
            ImageDataImpl impl;
            impl.mat = mat.clone();
            impl.width = width;
            impl.height = height;
            impl.channels = channels;
            impl.type = type;
            impl.ownership = RawMemoryOwnership::Owned;
            return impl;
        }

        bool empty() const { return mat.empty(); }
        size_t element_count() const { return mat.total(); }
        size_t element_bytes() const { return mat.elemSize(); }
        size_t bytes() const { return element_count() * element_bytes(); }
        const uint8_t* data() const { return mat.data; }
        uint8_t* data() { return mat.data; }
    };

    ImageData::ImageData(): impl_(std::make_unique<ImageDataImpl>()) {
    }

    ImageData::ImageData(int width, int height, const MdImageType type)
        : impl_(std::make_unique<ImageDataImpl>(width, height, type)) {
    }


    ImageData::ImageData(const cv::Mat& mat):
        impl_(std::make_unique<ImageDataImpl>(mat, true)) {
    }

    ImageData::ImageData(cv::Mat&& mat):
        impl_(std::make_unique<ImageDataImpl>(std::move(mat))) {
    }

    ImageData::~ImageData() = default;

    ImageData::ImageData(const ImageData& other)
        : impl_(other.impl_
                    ? std::make_unique<ImageDataImpl>(*other.impl_)
                    : std::make_unique<ImageDataImpl>()) {
    }

    ImageData& ImageData::operator=(const ImageData& other) {
        if (this != &other) {
            if (other.impl_) {
                impl_ = std::make_unique<ImageDataImpl>(*other.impl_);
            }
            else {
                impl_ = std::make_unique<ImageDataImpl>();
            }
        }
        return *this;
    }

    ImageData::ImageData(ImageData&& other) noexcept
        : impl_(std::move(other.impl_)) {
        other.impl_ = std::make_unique<ImageDataImpl>();
    }

    ImageData& ImageData::operator=(ImageData&& other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
            other.impl_ = std::make_unique<ImageDataImpl>();
        }
        return *this;
    }


    ImageData ImageData::clone() const {
        ImageData result;
        if (impl_) {
            result.impl_ = std::make_unique<ImageDataImpl>(impl_->clone());
        }
        return result;
    }


    ImageData ImageData::from_raw(unsigned char* data,
                                  const int width,
                                  const int height,
                                  const MdImageType type,
                                  const bool copy) {
        const int ocv_type = md_image_type_to_ocv_type(type);
        cv::Mat tmp_mat;
        if (ocv_type > 0) {
            tmp_mat = cv::Mat(height, width, ocv_type, data);
        }
        else if (type == MdImageType::I420 || type == MdImageType::NV12 || type == MdImageType::NV21) {
            tmp_mat = cv::Mat(height + height / 2, width, CV_8UC1, data);
        }
        else {
            MD_LOG_ERROR << "Invalid MdImageType format: " << md_image_type_to_string(type) << std::endl;
            return ImageData();
        }
        if (copy) {
            return ImageData(tmp_mat);
        }
        return ImageData(std::move(tmp_mat));
    }


    int ImageData::width() const { return impl_ ? impl_->width : 0; }
    int ImageData::height() const { return impl_ ? impl_->height : 0; }
    int ImageData::channels() const { return impl_ ? impl_->channels : 0; }
    MdImageType ImageData::type() const { return impl_ ? impl_->type : MdImageType::PKG_BGR_U8; }
    bool ImageData::empty() const { return !impl_ || impl_->empty(); }


    size_t ImageData::element_count() const { return impl_ ? impl_->element_count() : 0; }
    size_t ImageData::element_bytes() const { return impl_ ? impl_->element_bytes() : 0; }
    size_t ImageData::bytes() const { return impl_ ? impl_->bytes() : 0; }
    const uint8_t* ImageData::data() const { return impl_ ? impl_->data() : nullptr; }
    uint8_t* ImageData::data() { return impl_ ? impl_->data() : nullptr; }


    ImageData& ImageData::rotate(const RotateFlags flag) {
        if (!impl_ || impl_->empty()) {
            return *this;
        }
        cv::rotate(impl_->mat, impl_->mat, flag);
        impl_->width = impl_->mat.cols;
        impl_->height = impl_->mat.rows;
        return *this;
    }


    ImageData ImageData::crop(const Rect2f& rect) const {
        if (!impl_ || impl_->empty()) {
            return ImageData();
        }
        cv::Rect2f cv_rect(rect.x, rect.y, rect.width, rect.height);
        // 确保矩形在图像范围内
        cv_rect = cv_rect & cv::Rect2f(0, 0, impl_->width, impl_->height);
        if (cv_rect.width <= 0 || cv_rect.height <= 0) {
            return ImageData();
        }
        cv::Mat cropped = impl_->mat(cv_rect).clone();
        return ImageData(std::move(cropped));
    }

    ImageData ImageData::rotate_crop(std::array<float, 8> box) const {
        if (!impl_ || impl_->empty()) {
            return ImageData();
        }
        cv::Mat image;
        impl_->mat.copyTo(image);
        std::vector<std::vector<float>> points;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> tmp;
            tmp.push_back(box[2 * i]);
            tmp.push_back(box[2 * i + 1]);
            points.push_back(tmp);
        }
        float x_collect[4] = {box[0], box[2], box[4], box[6]};
        float y_collect[4] = {box[1], box[3], box[5], box[7]};
        float left = *std::min_element(x_collect, x_collect + 4);
        float right = *std::max_element(x_collect, x_collect + 4);
        float top = *std::min_element(y_collect, y_collect + 4);
        float bottom = *std::max_element(y_collect, y_collect + 4);
        cv::Mat img_crop;
        image(cv::Rect2f(left, top, right - left, bottom - top)).copyTo(img_crop);
        for (auto& point : points) {
            point[0] -= left;
            point[1] -= top;
        }

        float img_crop_width = sqrt(pow(points[0][0] - points[1][0], 2) +
            pow(points[0][1] - points[1][1], 2));
        float img_crop_height = sqrt(pow(points[0][0] - points[3][0], 2) +
            pow(points[0][1] - points[3][1], 2));

        cv::Point2f pts_std[4];
        pts_std[0] = cv::Point2f(0., 0.);
        pts_std[1] = cv::Point2f(img_crop_width, 0.);
        pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
        pts_std[3] = cv::Point2f(0.f, img_crop_height);

        cv::Point2f pointsf[4];
        pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
        pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
        pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
        pointsf[3] = cv::Point2f(points[3][0], points[3][1]);
        cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);
        cv::Mat dst_img;
        cv::warpPerspective(img_crop, dst_img, M,
                            cv::Size(img_crop_width, img_crop_height),
                            cv::BORDER_REPLICATE);

        if (dst_img.rows >= dst_img.cols * 1.5) {
            cv::transpose(dst_img, dst_img);
            cv::flip(dst_img, dst_img, 0);
        }
        return ImageData(std::move(dst_img));
    }


    ImageData ImageData::resize(int width, int height) const {
        if (!impl_ || impl_->empty() || width <= 0 || height <= 0) {
            return ImageData();
        }

        cv::Mat resized;
        cv::resize(impl_->mat, resized, cv::Size(width, height));
        return ImageData(std::move(resized));
    }

    ImageData ImageData::cast(const std::string& dtype, bool scale) const {
        if (!impl_ || impl_->empty()) {
            return ImageData();
        }
        const float scale_factor = scale ? 1.0f / 255.0f : 1.0f;
        cv::Mat converted;
        if (dtype == "float" || dtype == "float32" || dtype == "fp32") {
            if (impl_->mat.type() != CV_32FC(impl_->channels)) {
                impl_->mat.convertTo(converted, CV_32FC(impl_->channels), scale_factor);
            }
            else {
                converted = impl_->mat.clone();
            }
        }
        else if (dtype == "float16" || dtype == "fp16") {
            if (impl_->mat.type() != CV_16FC(impl_->channels)) {
                impl_->mat.convertTo(converted, CV_16FC(impl_->channels), scale_factor);
            }
            else {
                converted = impl_->mat.clone();
            }
        }
        else if (dtype == "double" || dtype == "float64" || dtype == "fp64") {
            if (impl_->mat.type() != CV_64FC(impl_->channels)) {
                impl_->mat.convertTo(converted, CV_64FC(impl_->channels), scale_factor);
            }
            else {
                converted = impl_->mat.clone();
            }
        }
        else {
            MD_LOG_WARN << "Cast not supported for " << dtype << ", returning original image." << std::endl;
            return clone();
        }
        return ImageData(std::move(converted));
    }

    ImageData ImageData::pad(const int top, const int bottom, const int left, const int right,
                             const float value) const {
        if (!impl_ || impl_->empty()) {
            return ImageData();
        }
        cv::Scalar padding_scalar;
        switch (channels()) {
        case 1: padding_scalar = cv::Scalar(value);
            break;
        case 3: padding_scalar = cv::Scalar(value, value, value);
            break;
        case 4: padding_scalar = cv::Scalar(value, value, value, value);
            break;
        default: {
            MD_LOG_ERROR << "Unsupported image channels: " << channels() << std::endl;
            return ImageData();
        }
        }
        cv::Mat padded;
        cv::copyMakeBorder(impl_->mat, padded, top, bottom, left, right, cv::BORDER_CONSTANT, padding_scalar);
        return ImageData(std::move(padded));
    }

    ImageData ImageData::convert(const std::vector<float>& alpha, const std::vector<float>& beta) const {
        if (channels() != 3 || channels() != alpha.size() || channels() != beta.size()) {
            MD_LOG_ERROR << "channels must be 3 and alpha/beta size must be 3" << std::endl;
            return ImageData();
        }
        std::vector<cv::Mat> split_im;
        cv::split(impl_->mat, split_im);
        for (int c = 0; c < impl_->mat.channels(); c++) {
            split_im[c].convertTo(split_im[c], CV_32FC1, alpha[c], beta[c]);
        }
        cv::Mat tmp_mat;
        cv::merge(split_im, tmp_mat);
        return ImageData(std::move(tmp_mat));
    }

    ImageData ImageData::normalize(const std::vector<float>& mean,
                                   const std::vector<float>& std,
                                   const bool scale,
                                   const bool swap_rb) const {
        if (channels() != 3 || channels() != mean.size() || channels() != std.size()) {
            MD_LOG_ERROR << "channels must be 3 and mean/std size must be 3" << std::endl;
            return ImageData();
        }
        std::vector<float> alpha;
        std::vector<float> beta;
        for (int i = 0; i < channels(); i++) {
            auto _alpha = 1.0f / std[i];
            _alpha = scale ? _alpha / 255.0f : _alpha;
            auto _beta = -mean[i] / std[i];
            alpha.push_back(_alpha);
            beta.push_back(_beta);
        }
        std::vector<cv::Mat> split_im;
        cv::split(impl_->mat, split_im);
        if (swap_rb) std::swap(split_im[0], split_im[2]);
        for (int c = 0; c < impl_->mat.channels(); c++) {
            split_im[c].convertTo(split_im[c], CV_32FC1, alpha[c], beta[c]);
        }
        cv::Mat tmp_mat;
        cv::merge(split_im, tmp_mat);
        return ImageData(std::move(tmp_mat));
    }

    ImageData ImageData::letter_box(const std::vector<int>& dst_size, const float padding_value) const {
        const float src_w = static_cast<float>(width());
        const float src_h = static_cast<float>(height());
        const float dst_w = static_cast<float>(dst_size[0]);
        const float dst_h = static_cast<float>(dst_size[1]);
        const float scale = std::min(dst_h / src_h, dst_w / src_w);
        const float resize_w = src_w * scale;
        const float resize_h = src_h * scale;
        const float pad_w = (dst_w - resize_w) * 0.5f;
        const float pad_h = (dst_h - resize_h) * 0.5f;

        cv::Scalar padding_scalar;
        switch (channels()) {
        case 1: padding_scalar = cv::Scalar(padding_value);
            break;
        case 3: padding_scalar = cv::Scalar(padding_value, padding_value, padding_value);
            break;
        case 4: padding_scalar = cv::Scalar(padding_value, padding_value, padding_value, padding_value);
            break;
        default: {
            MD_LOG_ERROR << "Unsupported image channels." << std::endl;
            return ImageData();
        }
        }
        cv::Mat tmp_image(dst_size[1], dst_size[0], impl_->mat.type(), padding_scalar);
        cv::Mat roi = tmp_image(cv::Rect(pad_w, pad_h, resize_w, resize_h));
        cv::resize(impl_->mat, roi, cv::Size(resize_w, resize_h));
        return ImageData(std::move(tmp_image));
    }

    [[nodiscard]] ImageData ImageData::center_crop(const std::vector<int>& dst_size) const {
        if (!impl_ || impl_->empty() || dst_size.size() != 2) {
            return ImageData();
        }
        if (width() < dst_size[0] || height() < dst_size[1]) {
            throw std::invalid_argument("ImageData::center_crop: dst_size must be smaller than image size.");
        }
        const int offset_x = (width() - dst_size[0]) / 2;
        const int offset_y = (height() - dst_size[1]) / 2;
        const Rect2f crop_roi(offset_x, offset_y, dst_size[0], dst_size[1]);
        return crop(crop_roi);
    }

    ImageData ImageData::permute() const {
        return cvt_color(*this, ColorConvertType::CVT_PA_RGB2PL_RGB);
    }

    ImageData ImageData::fuse_normalize_and_permute(const std::vector<float>& mean,
                                                    const std::vector<float>& std,
                                                    const bool scale) const {
        if (channels() != 3 || channels() != mean.size() || channels() != std.size()) {
            MD_LOG_ERROR << "channels must be 3 and mean/std size must be 3" << std::endl;
            return ImageData();
        }
        std::vector<float> alpha;
        std::vector<float> beta;
        for (int i = 0; i < channels(); i++) {
            auto _alpha = 1.0f / std[i];
            _alpha = scale ? _alpha / 255.0f : _alpha;
            auto _beta = -mean[i] / std[i];
            alpha.push_back(_alpha);
            beta.push_back(_beta);
        }
        cv::Mat chw_image(channels(), height() * width(), CV_32FC1);
        std::vector<cv::Mat> split_image;
        cv::split(impl_->mat, split_image);
        std::swap(split_image[0], split_image[2]);
        for (int c = 0; c < channels(); ++c) {
            // 转换为浮点并归一化
            cv::Mat channel_float;
            split_image[c].convertTo(channel_float, CV_32FC1, alpha[c], beta[c]);
            // 展平为一行
            channel_float = channel_float.reshape(1, 1);
            channel_float.copyTo(chw_image.row(c));
        }

        ImageData dst_image;
        dst_image.impl_ = std::make_unique<ImageDataImpl>();
        dst_image.impl_->mat = chw_image.reshape(1, {channels(), height(), width()});
        dst_image.impl_->width = width();
        dst_image.impl_->height = height();
        dst_image.impl_->channels = channels();
        dst_image.impl_->type = MdImageType::PLA_RGB_F32;
        return dst_image;
    }

    ImageData ImageData::fuse_convert_and_permute(const std::vector<float>& alpha,
                                                  const std::vector<float>& beta) const {
        if (channels() != 3 || channels() != alpha.size() || channels() != beta.size()) {
            MD_LOG_ERROR << "channels must be 3 and alpha/beta size must be 3" << std::endl;
            return ImageData();
        }
        const cv::Mat chw_image(channels(), height() * width(), CV_32FC1);
        std::vector<cv::Mat> split_image;
        cv::split(impl_->mat, split_image);
        std::swap(split_image[0], split_image[2]);
        for (int c = 0; c < channels(); ++c) {
            cv::Mat channel_float;
            split_image[c].convertTo(channel_float, CV_32FC1, alpha[c], beta[c]);
            channel_float = channel_float.reshape(1, 1);
            channel_float.copyTo(chw_image.row(c));
        }
        ImageData dst_image;
        dst_image.impl_ = std::make_unique<ImageDataImpl>();
        dst_image.impl_->mat = chw_image.reshape(1, {channels(), height(), width()});
        dst_image.impl_->width = width();
        dst_image.impl_->height = height();
        dst_image.impl_->channels = 3;
        dst_image.impl_->type = MdImageType::PLA_RGB_F32;
        return dst_image;
    }


    ImageData ImageData::cvt_color(const ImageData& image, ColorConvertType type) {
        if (!image.impl_ || image.impl_->empty()) {
            return ImageData();
        }
        const auto ocv_type = md_color_convert_type_to_ocv_color_convert_type(type);
        if (ocv_type > 0) {
            cv::Mat converted;
            cv::cvtColor(image.impl_->mat, converted, ocv_type);
            return ImageData(std::move(converted));
        }
        if (type == ColorConvertType::CVT_PA_BGR2PL_BGR || type == ColorConvertType::CVT_PA_RGB2PL_RGB) {
            ImageData dst_image;
            dst_image.impl_ = std::make_unique<ImageDataImpl>();
            const int single_channel_type = CV_MAKETYPE(image.impl_->mat.depth(), 1);
            cv::Mat chw_image(image.channels(), image.height() * image.width(), single_channel_type);
            std::vector<cv::Mat> split_image;
            cv::split(image.impl_->mat, split_image);
            for (int i = 0; i < split_image.size(); i++) {
                split_image[i] = split_image[i].reshape(1, 1);
                split_image[i].copyTo(chw_image.row(i));
            }
            dst_image.impl_->mat = chw_image.reshape(1, {image.channels(), image.height(), image.width()});
            dst_image.impl_->width = image.width();
            dst_image.impl_->height = image.height();
            dst_image.impl_->channels = image.channels();
            dst_image.impl_->type = image.impl_->mat.depth() == CV_8U
                                        ? MdImageType::PLA_BGR_U8
                                        : MdImageType::PLA_BGR_F32;
            return dst_image;
        }
        if (type == ColorConvertType::CVT_PL_BGR2PA_BGR || type == ColorConvertType::CVT_PL_RGB2PA_RGB) {
            // valid chw format
            if (image.type() != MdImageType::PLA_BGR_U8 && image.type() != MdImageType::PLA_BGR_F32
                && image.type() != MdImageType::PLA_RGB_U8 && image.type() != MdImageType::PLA_RGB_F32) {
                throw std::runtime_error("Invalid PL_BGR format: expected Planar layout");
            }
            ImageData dst_image;
            dst_image.impl_ = std::make_unique<ImageDataImpl>();

            // 1 channel per row, total rows equal to channels
            cv::Mat planar_image = image.impl_->mat.reshape(1, image.channels());
            // 2. Split the planar image into separate channel matrices.
            std::vector<cv::Mat> split_images(image.channels());
            for (int i = 0; i < image.channels(); ++i) {
                split_images[i] = planar_image.row(i).reshape(1, image.height()); // reshape each row back to H x W
            }
            // 3. Merge these channel matrices into a single HWC image.
            cv::Mat hwc_image;
            cv::merge(split_images, hwc_image);

            dst_image.impl_->mat = hwc_image;
            dst_image.impl_->width = image.width();
            dst_image.impl_->height = image.height();
            dst_image.impl_->channels = image.channels();
            dst_image.impl_->type = hwc_image.depth() == CV_8U
                                        ? MdImageType::PKG_BGR_U8
                                        : hwc_image.depth() == CV_32F
                                        ? MdImageType::PKG_BGR_F32
                                        : MdImageType::PKG_BGR_F64;
            return dst_image;
        }
        throw std::runtime_error("Unsupported color conversion type");
    }

    void ImageData::images_to_tensor(std::vector<ImageData> images, Tensor* tensor) {
        if (images.empty()) {
            MD_LOG_ERROR << "images is empty" << std::endl;
            return;
        }

        const int n = images.size();
        const int c = images[0].channels();
        const int h = images[0].height();
        const int w = images[0].width();

        for (auto& img : images) {
            if (img.channels() != c || img.width() != w || img.height() != h) {
                MD_LOG_ERROR << "images shape is not equal" << std::endl;
                return;
            }
        }
        const size_t bytes = images[0].bytes();
        const std::vector<int64_t> shape = {n, c, h, w};
        const auto dtype = utils::md_image_dtype_to_md_dtype(images[0].type());
        tensor->allocate(shape, dtype);
        for (size_t i = 0; i < images.size(); ++i) {
            auto* p = static_cast<uint8_t*>(tensor->data());
            std::memcpy(p + i * bytes, images[i].data(), bytes);
        }
    }

    void ImageData::to_tensor(Tensor* tensor, bool copy) {
        if (!impl_ || impl_->empty()) {
            MD_LOG_ERROR << "Image is empty" << std::endl;
            return;
        }
        const auto dtype = utils::md_image_dtype_to_md_dtype(type());
        const std::vector<int64_t> shape = {channels(), height(), width()};
        if (copy) {
            const size_t num_bytes = bytes();
            tensor->allocate(shape, dtype);
            if (num_bytes != tensor->byte_size()) {
                MD_LOG_ERROR << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = " << tensor->byte_size()
                    << ", size of Mat = " << num_bytes << "." << std::endl;
            }
            memcpy(tensor->data(), data(), num_bytes);
        }
        else {
            tensor->from_external_memory(data(), shape, dtype);
        }
    }

    void ImageData::to_mat(cv::Mat& mat, const bool copy) const {
        if (!impl_ || impl_->empty()) {
            return;
        }
        if (copy) {
            // Shallow copy: share underlying data
            mat = impl_->mat.clone();
        }
        else {
            mat = impl_->mat;
        }
    }


    ImageData ImageData::imread(const std::string& filename) {
        cv::Mat mat = cv::imread(filename);
        if (mat.empty()) {
            MD_LOG_ERROR << "Failed to read image: " << filename << std::endl;
            return ImageData();
        }
        return ImageData(std::move(mat));
    }

    bool ImageData::imwrite(const std::string& filename) const {
        if (!impl_ || impl_->empty()) {
            MD_LOG_ERROR << "Cannot write empty image" << std::endl;
            return false;
        }
        return cv::imwrite(filename, impl_->mat);
    }

    // 显示图片
    void ImageData::imshow(const std::string& win_name) const {
        if (!impl_ || impl_->empty()) {
            MD_LOG_ERROR << "Cannot display empty image" << std::endl;
            return;
        }
        cv::imshow(win_name, impl_->mat);
        cv::waitKey(0);
    }
}
