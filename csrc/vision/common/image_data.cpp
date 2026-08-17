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
        ImageDataImpl() = default;
        ImageDataImpl(const ImageDataImpl& other) = default;
        ImageDataImpl& operator=(const ImageDataImpl& other) = default;
        ImageDataImpl(ImageDataImpl&& other) noexcept = default;
        ImageDataImpl& operator=(ImageDataImpl&& other) noexcept = default;

        // 平面描述：packed 格式 1 个平面（data），NV12/NV21 2 个平面（y, uv），I420 3 个平面（y, u, v）
        struct Plane { uint8_t* data = nullptr; int step = 0; };

        void refresh_meta() {
            if (device != Device::CPU) {
                width = height = channels = 0;
                element_count_ = element_bytes_ = bytes_ = 0;
                return;
            }
            // CPU 模式下 mat 仍是事实源，refresh 逻辑保持不变
            if (mat.empty()) { width = height = channels = 0; element_count_ = element_bytes_ = bytes_ = 0; return; }
            if (is_planar_type(type) && mat.dims >= 3) {
                channels = static_cast<int>(mat.size[0]);
                height = static_cast<int>(mat.size[1]);
                width = static_cast<int>(mat.size[2]);
            } else {
                width = mat.cols; height = mat.rows;
                channels = is_planar_type(type) ? static_cast<int>(mat.size[0]) : mat.channels();
            }
            element_count_ = mat.total();
            element_bytes_ = mat.elemSize();
            bytes_ = element_count_ * element_bytes_;
        }

        bool empty() const {
            if (device != Device::CPU) return planes.empty() || !planes[0].data;
            return mat.empty();
        }
        size_t element_count() const { return element_count_; }
        size_t element_bytes() const { return element_bytes_; }
        size_t bytes() const { return bytes_; }
        const uint8_t* data() const { return device == Device::CPU && !mat.empty() ? mat.data : (planes.empty() ? nullptr : planes[0].data); }
        uint8_t* data() { return device == Device::CPU && !mat.empty() ? mat.data : (planes.empty() ? nullptr : planes[0].data); }

        cv::Mat mat;                       // CPU 模式：事实数据源；设备模式：空
        std::vector<Plane> planes;         // 设备/裸指针模式：平面指针+步长；CPU 模式也填充（data 指向 mat.data）
        Device device = Device::CPU;
        int width = 0;
        int height = 0;
        int channels = 0;
        MdImageType type = MdImageType::PKG_BGR_U8;
        size_t element_count_ = 0;
        size_t element_bytes_ = 0;
        size_t bytes_ = 0;
    };

    ImageData::ImageData(const int width, const int height, const MdImageType type)
        : impl_(std::make_shared<ImageDataImpl>()) {
        impl_->type = type;
        const int ocv_type = md_image_type_to_ocv_type(type);
        impl_->mat = cv::Mat(height, width, ocv_type);
        // 构造后同步平面描述（CPU 模式 data 指向 mat.data）
        if (!impl_->mat.empty()) {
            impl_->planes.clear();
            if (impl_->type == MdImageType::NV12 || impl_->type == MdImageType::NV21) {
                // Y 平面 h 行，UV 平面 h/2 行（沿用 from_raw 的单 buffer 布局：mat 为 (h+h/2, w)）
                impl_->planes.push_back({impl_->mat.data, width});
                impl_->planes.push_back({impl_->mat.data + static_cast<size_t>(height) * width, width});
            } else {
                impl_->planes.push_back({impl_->mat.data, static_cast<int>(impl_->mat.step)});
            }
        }
        impl_->refresh_meta();
    }


    ImageData::ImageData(const cv::Mat& mat) :
        impl_(std::make_shared<ImageDataImpl>()) {
        impl_->mat = mat;
        impl_->type = md_image_type_from_ocv_type(mat.type());
        // 构造后同步平面描述（CPU 模式 data 指向 mat.data）
        if (!impl_->mat.empty()) {
            impl_->planes.clear();
            if (impl_->type == MdImageType::NV12 || impl_->type == MdImageType::NV21) {
                // Y 平面 h 行，UV 平面 h/2 行（沿用 from_raw 的单 buffer 布局：mat 为 (h+h/2, w)）
                const int w = impl_->mat.cols;
                const int h = 2 * impl_->mat.rows / 3;
                impl_->planes.push_back({impl_->mat.data, w});
                impl_->planes.push_back({impl_->mat.data + static_cast<size_t>(h) * w, w});
            } else {
                impl_->planes.push_back({impl_->mat.data, static_cast<int>(impl_->mat.step)});
            }
        }
        impl_->refresh_meta();
    }

    ImageData::ImageData(cv::Mat&& mat) :
        impl_(std::make_shared<ImageDataImpl>()) {
        impl_->mat = std::move(mat);
        impl_->type = md_image_type_from_ocv_type(impl_->mat.type());
        // 构造后同步平面描述（CPU 模式 data 指向 mat.data）
        if (!impl_->mat.empty()) {
            impl_->planes.clear();
            if (impl_->type == MdImageType::NV12 || impl_->type == MdImageType::NV21) {
                // Y 平面 h 行，UV 平面 h/2 行（沿用 from_raw 的单 buffer 布局：mat 为 (h+h/2, w)）
                const int w = impl_->mat.cols;
                const int h = 2 * impl_->mat.rows / 3;
                impl_->planes.push_back({impl_->mat.data, w});
                impl_->planes.push_back({impl_->mat.data + static_cast<size_t>(h) * w, w});
            } else {
                impl_->planes.push_back({impl_->mat.data, static_cast<int>(impl_->mat.step)});
            }
        }
        impl_->refresh_meta();
    }

    ImageData ImageData::clone() const {
        ImageData result;
        if (impl_) {
            result.impl_ = std::make_shared<ImageDataImpl>();
            result.impl_->mat = impl_->mat.clone();
            result.impl_->type = impl_->type;
            // 构造后同步平面描述（CPU 模式 data 指向 mat.data）
            if (!result.impl_->mat.empty()) {
                result.impl_->planes.clear();
                if (result.impl_->type == MdImageType::NV12 || result.impl_->type == MdImageType::NV21) {
                    // Y 平面 h 行，UV 平面 h/2 行（沿用 from_raw 的单 buffer 布局：mat 为 (h+h/2, w)）
                    const int w = result.impl_->mat.cols;
                    const int h = 2 * result.impl_->mat.rows / 3;
                    result.impl_->planes.push_back({result.impl_->mat.data, w});
                    result.impl_->planes.push_back({result.impl_->mat.data + static_cast<size_t>(h) * w, w});
                } else {
                    result.impl_->planes.push_back({result.impl_->mat.data, static_cast<int>(result.impl_->mat.step)});
                }
            }
            result.impl_->refresh_meta();
        }
        return result;
    }


    ImageData ImageData::from_raw(unsigned char* data,
                                  const int width,
                                  const int height,
                                  const MdImageType type,
                                  const bool copy) {
        if (!data || width <= 0 || height <= 0) {
            MD_LOG_ERROR << "Invalid parameters for from_raw" << std::endl;
            return ImageData();
        }
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
            return ImageData(tmp_mat.clone());
        }
        return ImageData(tmp_mat);
    }


    int ImageData::width() const { return impl_ ? impl_->width : 0; }
    int ImageData::height() const { return impl_ ? impl_->height : 0; }
    int ImageData::channels() const { return impl_ ? impl_->channels : 0; }
    MdImageType ImageData::type() const { return impl_ ? impl_->type : MdImageType::PKG_BGR_U8; }
    bool ImageData::empty() const { return !impl_ || impl_->empty(); }

    bool ImageData::is_shared_with(const ImageData& other) const {
        return impl_ && other.impl_ && impl_.get() == other.impl_.get();
    }

    size_t ImageData::element_count() const { return impl_ ? impl_->element_count() : 0; }
    size_t ImageData::element_bytes() const { return impl_ ? impl_->element_bytes() : 0; }
    size_t ImageData::bytes() const { return impl_ ? impl_->bytes() : 0; }
    const uint8_t* ImageData::data() const { return impl_ ? impl_->data() : nullptr; }
    uint8_t* ImageData::data() { return impl_ ? impl_->data() : nullptr; }

    Device ImageData::device() const { return impl_ ? impl_->device : Device::CPU; }
    const uint8_t* ImageData::y() const {
        if (!impl_ || impl_->planes.size() < 1) return nullptr;
        return impl_->planes[0].data;
    }
    const uint8_t* ImageData::uv() const {
        if (!impl_ || impl_->planes.size() < 2) return nullptr;
        return impl_->planes[1].data;
    }
    int ImageData::step_y() const { return impl_ && !impl_->planes.empty() ? impl_->planes[0].step : 0; }
    int ImageData::step_uv() const { return impl_ && impl_->planes.size() >= 2 ? impl_->planes[1].step : 0; }
    size_t ImageData::plane_count() const { return impl_ ? impl_->planes.size() : 0; }

    ImageData ImageData::from_device_planes(uint8_t* y, uint8_t* uv, int w, int h,
                                            int step_y, int step_uv, Device device) {
        if (!y || w <= 0 || h <= 0) {
            MD_LOG_ERROR << "from_device_planes: invalid parameters" << std::endl;
            return ImageData();
        }
        ImageData img;
        img.impl_ = std::make_shared<ImageDataImpl>();
        img.impl_->device = device;
        img.impl_->type = MdImageType::NV12;
        img.impl_->width = w;
        img.impl_->height = h;
        img.impl_->channels = 1;
        img.impl_->planes.push_back({y, step_y > 0 ? step_y : w});
        if (uv) img.impl_->planes.push_back({uv, step_uv > 0 ? step_uv : w});
        img.impl_->element_count_ = static_cast<size_t>(w) * h;
        img.impl_->element_bytes_ = 1;
        img.impl_->bytes_ = static_cast<size_t>(w) * h * 3 / 2;
        return img;
    }


    ImageData& ImageData::rotate(const RotateFlags flag) {
        if (!impl_ || impl_->empty()) {
            return *this;
        }
        cv::rotate(impl_->mat, impl_->mat, flag);
        impl_->refresh_meta();
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
        // 直接在原图上取 ROI，避免整图拷贝（密集文本页每行一次整图 copy 开销很大）
        cv::Rect roi(std::max(0, static_cast<int>(left)), std::max(0, static_cast<int>(top)),
                     std::max(1, static_cast<int>(right - left)), std::max(1, static_cast<int>(bottom - top)));
        cv::Mat img_crop;
        impl_->mat(roi & cv::Rect(0, 0, impl_->mat.cols, impl_->mat.rows)).copyTo(img_crop);
        for (auto& point : points) {
            point[0] -= left;
            point[1] -= top;
        }

        const float img_crop_width = sqrt(pow(points[0][0] - points[1][0], 2) +
            pow(points[0][1] - points[1][1], 2));
        const float img_crop_height = sqrt(pow(points[0][0] - points[3][0], 2) +
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

    ImageData ImageData::cvt_color(const ImageData& image, const ColorConvertType type) {
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
            dst_image.impl_ = std::make_shared<ImageDataImpl>();
            const int single_channel_type = CV_MAKETYPE(image.impl_->mat.depth(), 1);
            cv::Mat chw_image(image.channels(), image.height() * image.width(), single_channel_type);
            std::vector<cv::Mat> split_image;
            cv::split(image.impl_->mat, split_image);
            for (int i = 0; i < split_image.size(); i++) {
                split_image[i] = split_image[i].reshape(1, 1);
                split_image[i].copyTo(chw_image.row(i));
            }
            dst_image.impl_->mat = chw_image.reshape(1, {image.channels(), image.height(), image.width()});
            dst_image.impl_->type = image.impl_->mat.depth() == CV_8U
                                        ? MdImageType::PLA_BGR_U8
                                        : MdImageType::PLA_BGR_F32;
            dst_image.impl_->refresh_meta();
            return dst_image;
        }
        if (type == ColorConvertType::CVT_PL_BGR2PA_BGR || type == ColorConvertType::CVT_PL_RGB2PA_RGB) {
            // valid chw format
            if (image.type() != MdImageType::PLA_BGR_U8 && image.type() != MdImageType::PLA_BGR_F32
                && image.type() != MdImageType::PLA_RGB_U8 && image.type() != MdImageType::PLA_RGB_F32) {
                throw std::runtime_error("Invalid PL_BGR format: expected Planar layout");
            }
            ImageData dst_image;
            dst_image.impl_ = std::make_shared<ImageDataImpl>();

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
            dst_image.impl_->type = hwc_image.depth() == CV_8U
                                        ? MdImageType::PKG_BGR_U8
                                        : hwc_image.depth() == CV_32F
                                        ? MdImageType::PKG_BGR_F32
                                        : MdImageType::PKG_BGR_F64;
            dst_image.impl_->refresh_meta();
            return dst_image;
        }
        throw std::runtime_error("Unsupported color conversion type");
    }

    void ImageData::images_to_tensor(const std::vector<ImageData>& images, Tensor* tensor) {
        if (images.empty() || !tensor) {
            MD_LOG_ERROR << "images is empty or tensor is null" << std::endl;
            return;
        }
        const int n = static_cast<int>(images.size());
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
        // allocate 复用逻辑：shape/dtype/device 不变时复用已有 MemoryBlock，避免每帧重新分配
        tensor->allocate(shape, dtype);
        uint8_t* base = static_cast<uint8_t*>(tensor->data());
        for (size_t i = 0; i < images.size(); ++i) {
            const uint8_t* src = images[i].data();
            if (src) {
                std::memcpy(base + i * bytes, src, bytes);
            }
        }
    }

    void ImageData::to_tensor(Tensor* tensor, const bool copy) {
        if (!impl_ || impl_->empty()) {
            MD_LOG_ERROR << "Image is empty" << std::endl;
            return;
        }
        if (!tensor) {
            MD_LOG_ERROR << "Tensor pointer is null" << std::endl;
            return;
        }
        const auto dtype = utils::md_image_dtype_to_md_dtype(type());
        const std::vector<int64_t> shape = {channels(), height(), width()};
        if (copy) {
            const size_t num_bytes = bytes();
            // allocate 复用逻辑：shape/dtype/device 不变时复用已有 MemoryBlock
            tensor->allocate(shape, dtype);
            if (num_bytes != tensor->byte_size()) {
                MD_LOG_ERROR << "While copy Mat to Tensor, requires the memory size be same, "
                    "but now size of Tensor = " << tensor->byte_size()
                    << ", size of Mat = " << num_bytes << "." << std::endl;
                return;
            }
            if (data() && tensor->data()) {
                memcpy(tensor->data(), data(), num_bytes);
            }
        }
        else {
            // 零拷贝：共享外部内存，不复制
            tensor->from_external_memory(data(), shape, dtype);
        }
    }

    void ImageData::to_mat(cv::Mat& mat, const bool copy) const {
        if (!impl_ || impl_->empty()) {
            return;
        }
        if (copy) {
            // Deep copy: clone the underlying data
            mat = impl_->mat.clone();
        }
        else {
            // Shallow copy: share underlying data
            mat = impl_->mat;
        }
    }


    std::vector<uint8_t> ImageData::imencode(const ImageData& image, const std::string& ext) {
        std::vector<uint8_t> buf;
        if (image.empty()) {
            MD_LOG_ERROR << "Cannot encode empty image" << std::endl;
            return buf;
        }
        cv::imencode(ext, image.impl_->mat, buf);
        return buf;
    }

    ImageData ImageData::imdecode(const std::vector<uint8_t>& buf) {
        cv::Mat mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
        if (mat.empty()) return ImageData();
        return ImageData(std::move(mat));
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