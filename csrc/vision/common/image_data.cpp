//
// Created by aichao on 2025/7/18.
//

#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/convert.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include <opencv2/opencv.hpp>


namespace modeldeploy::vision {
    // thread_local 错误通道
    static thread_local std::string g_last_error_msg;

    // 置错误（成功返回 true 的操作不调用；失败时写入含 op 名的可读信息）
    static void set_last_error(const std::string& msg) {
        g_last_error_msg = msg;
    }

    // 按设备分派到对应预处理后端（CPU 由 CpuProcessorBackend 实现；设备帧由各自后端，
    // 未实现 → 返回 false，ImageData 置 last_error，不静默回退 CPU）
    static std::unique_ptr<VisionProcessorBackend> backend_for(Device d) {
        return create_processor_backend(d, d == Device::TPU ? Backend::SOPHGO : Backend::ORT, 0);
    }

    // 承载抽象：CPU=OpenCV、设备/借用=平面+RAII
    struct ImageDataStorage {
        Device device = Device::CPU;
        virtual ~ImageDataStorage() = default;
        [[nodiscard]] virtual Device dev() const = 0;
    };

    struct CpuStorage : ImageDataStorage {
        cv::Mat mat;                       // CPU 事实数据源
        [[nodiscard]] Device dev() const override { return Device::CPU; }
    };

    struct PlaneStorage : ImageDataStorage {   // 设备/借用平面
        std::vector<ImageData::Plane> planes;  // data 借用（外置，不拥有）
        std::shared_ptr<void> keeper;          // 借用源 RAII（可空），保活
        MdImageType fmt = MdImageType::NV12;
        int w = 0, h = 0, ch = 1;
        size_t nbytes = 0;
        [[nodiscard]] Device dev() const override { return device; }
    };

    class ImageDataImpl {
    public:
        MdImageType type = MdImageType::PKG_BGR_U8;
        int width = 0, height = 0, channels = 0;
        size_t element_count_ = 0, element_bytes_ = 0, bytes_ = 0;
        std::shared_ptr<ImageDataStorage> storage;

        // per-instance：CPU 借用平面包装 mat 的缓存 + 设备/借用(non-CPU) 的空 mat
        // （替代旧 shared static empty，避免并发线程在设备帧上写同一共享对象）
        mutable cv::Mat plane_mat_;

        // CPU 事实数据源 mat；CPU 借用平面 → 包装成借用 mat（不复制）；设备/借用(non-CPU) → 空 mat（保持旧 no-op 行为）
        cv::Mat& mat() {
            if (auto* cs = dynamic_cast<CpuStorage*>(storage.get())) return cs->mat;
            return materialize_plane_mat();
        }
        const cv::Mat& mat() const {
            if (auto* cs = dynamic_cast<CpuStorage*>(storage.get())) return cs->mat;
            return materialize_plane_mat();
        }

        // 仅 CPU 借用平面（from_bgr24 等）把外置平面包装成借用 mat；设备/借用(non-CPU) 返回空 mat（不 materialize）
        cv::Mat& materialize_plane_mat() const {
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) {
                if (ps->device == Device::CPU && !ps->planes.empty() && ps->planes[0].data && ps->w > 0 && ps->h > 0) {
                    const int ocv_type = md_image_type_to_ocv_type(ps->fmt);
                    if (ocv_type > 0) {
                        plane_mat_ = cv::Mat(ps->h, ps->w, ocv_type,
                                             const_cast<uint8_t*>(ps->planes[0].data));
                        return plane_mat_;
                    }
                }
            }
            plane_mat_ = cv::Mat();
            return plane_mat_;
        }

        bool is_cpu_plane() const {
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get()))
                return ps->device == Device::CPU;
            return false;
        }
        void set_cpu_plane_dims(int w, int h) {
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) {
                ps->w = w;
                ps->h = h;
            }
        }

        // 统一平面描述：PlaneStorage 直接用；CpuStorage 从 mat 派生
        std::vector<ImageData::Plane> planes() const {
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) return ps->planes;
            std::vector<ImageData::Plane> out;
            const cv::Mat& m = mat();
            if (m.empty()) return out;
            if (type == MdImageType::NV12 || type == MdImageType::NV21) {
                // Y 平面 h 行，UV 平面 h/2 行（沿用单 buffer 布局：mat 为 (h+h/2, w)）
                const int w = m.cols;
                const int h = 2 * m.rows / 3;
                out.push_back({m.data, w});
                out.push_back({m.data + static_cast<size_t>(h) * w, w});
            } else {
                out.push_back({m.data, static_cast<int>(m.step)});
            }
            return out;
        }

        void refresh_meta() {
            auto* cs = dynamic_cast<CpuStorage*>(storage.get());
            if (!cs) {
                // PlaneStorage：宽高通道已是构造时填好的真实值，不清零
                return;
            }
            if (cs->mat.empty()) {
                width = height = channels = 0;
                element_count_ = element_bytes_ = bytes_ = 0;
                return;
            }
            if (is_planar_type(type) && cs->mat.dims >= 3) {
                channels = static_cast<int>(cs->mat.size[0]);
                height = static_cast<int>(cs->mat.size[1]);
                width = static_cast<int>(cs->mat.size[2]);
            } else {
                width = cs->mat.cols; height = cs->mat.rows;
                channels = is_planar_type(type) ? static_cast<int>(cs->mat.size[0]) : cs->mat.channels();
            }
            element_count_ = cs->mat.total();
            element_bytes_ = cs->mat.elemSize();
            bytes_ = element_count_ * element_bytes_;
        }

        bool empty() const {
            if (auto* cs = dynamic_cast<CpuStorage*>(storage.get())) return cs->mat.empty();
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) return ps->planes.empty() || !ps->planes[0].data;
            return true;
        }

        const uint8_t* data() const {
            if (auto* cs = dynamic_cast<CpuStorage*>(storage.get())) return cs->mat.empty() ? nullptr : cs->mat.data;
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) return ps->planes.empty() ? nullptr : ps->planes[0].data;
            return nullptr;
        }
        uint8_t* data() {
            if (auto* cs = dynamic_cast<CpuStorage*>(storage.get())) return cs->mat.empty() ? nullptr : cs->mat.data;
            if (auto* ps = dynamic_cast<PlaneStorage*>(storage.get())) return ps->planes.empty() ? nullptr : const_cast<uint8_t*>(ps->planes[0].data);
            return nullptr;
        }
    };

    ImageData::ImageData(const int width, const int height, const MdImageType type)
        : impl_(std::make_shared<ImageDataImpl>()) {
        impl_->type = type;
        const int ocv_type = md_image_type_to_ocv_type(type);
        auto* cs = new CpuStorage();
        cs->mat = cv::Mat(height, width, ocv_type);
        impl_->storage.reset(cs);
        impl_->refresh_meta();
    }


    ImageData::ImageData(const cv::Mat& mat) :
        impl_(std::make_shared<ImageDataImpl>()) {
        auto* cs = new CpuStorage();
        cs->mat = mat;
        impl_->storage.reset(cs);
        impl_->type = md_image_type_from_ocv_type(mat.type());
        impl_->refresh_meta();
    }

    ImageData::ImageData(cv::Mat&& mat) :
        impl_(std::make_shared<ImageDataImpl>()) {
        auto* cs = new CpuStorage();
        cs->mat = std::move(mat);
        impl_->storage.reset(cs);
        impl_->type = md_image_type_from_ocv_type(cs->mat.type());
        impl_->refresh_meta();
    }

    ImageData ImageData::clone() const {
        ImageData result;
        if (impl_) {
            result.impl_ = std::make_shared<ImageDataImpl>();
            auto* cs = new CpuStorage();
            cs->mat = impl_->mat().clone();
            result.impl_->storage.reset(cs);
            result.impl_->type = impl_->type;
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
    MdImageType ImageData::format() const { return type(); }
    bool ImageData::empty() const { return !impl_ || impl_->empty(); }

    bool ImageData::is_shared_with(const ImageData& other) const {
        return impl_ && other.impl_ && impl_.get() == other.impl_.get();
    }

    size_t ImageData::element_count() const { return impl_ ? impl_->element_count_ : 0; }
    size_t ImageData::element_bytes() const { return impl_ ? impl_->element_bytes_ : 0; }
    size_t ImageData::bytes() const { return impl_ ? impl_->bytes_ : 0; }
    const uint8_t* ImageData::data() const { return impl_ ? impl_->data() : nullptr; }
    uint8_t* ImageData::data() { return impl_ ? impl_->data() : nullptr; }

    Device ImageData::device() const { return impl_ ? (impl_->storage ? impl_->storage->device : Device::CPU) : Device::CPU; }
    const uint8_t* ImageData::y() const {
        if (!impl_) return nullptr;
        const auto pl = impl_->planes();
        return pl.size() < 1 ? nullptr : pl[0].data;
    }
    const uint8_t* ImageData::uv() const {
        if (!impl_) return nullptr;
        const auto pl = impl_->planes();
        return pl.size() < 2 ? nullptr : pl[1].data;
    }
    int ImageData::step_y() const {
        if (!impl_) return 0;
        const auto pl = impl_->planes();
        return pl.empty() ? 0 : pl[0].step;
    }
    int ImageData::step_uv() const {
        if (!impl_) return 0;
        const auto pl = impl_->planes();
        return pl.size() < 2 ? 0 : pl[1].step;
    }
    size_t ImageData::plane_count() const { return impl_ ? impl_->planes().size() : 0; }
    ImageData::Plane ImageData::plane(size_t i) const {
        if (!impl_) return {};
        const auto pl = impl_->planes();
        return i < pl.size() ? pl[i] : Plane{};
    }

    ImageData ImageData::from_device_planes(uint8_t* y, uint8_t* uv, int w, int h,
                                            int step_y, int step_uv, Device device) {
        if (!y || w <= 0 || h <= 0) {
            MD_LOG_ERROR << "from_device_planes: invalid parameters" << std::endl;
            return ImageData();
        }
        ImageData img;
        img.impl_ = std::make_shared<ImageDataImpl>();
        auto* ps = new PlaneStorage();
        ps->device = device;
        ps->fmt = MdImageType::NV12;
        ps->w = w;
        ps->h = h;
        ps->ch = 1;
        ps->planes.push_back({y, step_y > 0 ? step_y : w});
        if (uv) ps->planes.push_back({uv, step_uv > 0 ? step_uv : w});
        ps->nbytes = static_cast<size_t>(w) * h * 3 / 2;
        img.impl_->storage.reset(ps);
        img.impl_->type = MdImageType::NV12;
        img.impl_->width = w;
        img.impl_->height = h;
        img.impl_->channels = 1;
        img.impl_->element_count_ = static_cast<size_t>(w) * h;
        img.impl_->element_bytes_ = 1;
        img.impl_->bytes_ = ps->nbytes;
        return img;
    }

    ImageData ImageData::from_bgr24(const uint8_t* bgr, int w, int h) {
        g_last_error_msg.clear();
        if (!bgr || w <= 0 || h <= 0) {
            g_last_error_msg = "from_bgr24: invalid parameters";
            return ImageData();
        }
        ImageData img;
        img.impl_ = std::make_shared<ImageDataImpl>();
        auto* ps = new PlaneStorage();
        ps->device = Device::CPU;
        ps->fmt = MdImageType::PKG_BGR_U8;
        ps->w = w;
        ps->h = h;
        ps->ch = 3;
        ps->planes.push_back({bgr, static_cast<int>(static_cast<size_t>(w) * 3)});
        ps->nbytes = static_cast<size_t>(w) * h * 3;
        img.impl_->storage.reset(ps);
        img.impl_->type = MdImageType::PKG_BGR_U8;
        img.impl_->width = w;
        img.impl_->height = h;
        img.impl_->channels = 3;
        img.impl_->element_count_ = static_cast<size_t>(w) * h;
        img.impl_->element_bytes_ = 3;
        img.impl_->bytes_ = ps->nbytes;
        return img;
    }

    bool ImageData::toCpu(ImageData* out) const {
        g_last_error_msg.clear();
        if (!out) {
            g_last_error_msg = "toCpu: null output";
            return false;
        }
        if (!impl_ || impl_->empty()) {
            g_last_error_msg = "toCpu: source image is empty";
            return false;
        }
        if (auto* cs = dynamic_cast<CpuStorage*>(impl_->storage.get())) {
            // 已是 CPU：浅 clone（共享 mat）
            out->impl_ = std::make_shared<ImageDataImpl>();
            auto* ncs = new CpuStorage();
            ncs->mat = cs->mat;
            out->impl_->storage.reset(ncs);
            out->impl_->type = impl_->type;
            out->impl_->width = impl_->width;
            out->impl_->height = impl_->height;
            out->impl_->channels = impl_->channels;
            out->impl_->element_count_ = impl_->element_count_;
            out->impl_->element_bytes_ = impl_->element_bytes_;
            out->impl_->bytes_ = impl_->bytes_;
            return true;
        }
        // 设备/借用平面 → CPU 深拷贝
        const auto src_planes = impl_->planes();
        auto* ncs = new CpuStorage();
        if (impl_->type == MdImageType::NV12 || impl_->type == MdImageType::NV21) {
            const int w = impl_->width;
            const int h = impl_->height;
            cv::Mat nv12(h + h / 2, w, CV_8UC1);
            uint8_t* dst = nv12.data;
            if (src_planes.size() >= 1 && src_planes[0].data) {
                for (int r = 0; r < h; ++r) {
                    std::memcpy(dst + static_cast<size_t>(r) * w,
                                src_planes[0].data + static_cast<size_t>(r) * src_planes[0].step, w);
                }
            }
            if (src_planes.size() >= 2 && src_planes[1].data) {
                for (int r = 0; r < h / 2; ++r) {
                    std::memcpy(dst + static_cast<size_t>(h) * w + static_cast<size_t>(r) * w,
                                src_planes[1].data + static_cast<size_t>(r) * src_planes[1].step, w);
                }
            }
            ncs->mat = nv12;
        } else {
            // 单平面 packed（如 from_bgr24）
            const int w = impl_->width;
            const int h = impl_->height;
            const int ch = impl_->channels > 0 ? impl_->channels : 1;
            cv::Mat packed(h, w, CV_MAKETYPE(CV_8U, ch));
            if (src_planes.size() >= 1 && src_planes[0].data) {
                const int step_src = src_planes[0].step > 0 ? src_planes[0].step : w * ch;
                const int step_dst = static_cast<int>(packed.step);
                for (int r = 0; r < h; ++r) {
                    std::memcpy(packed.data + static_cast<size_t>(r) * step_dst,
                                src_planes[0].data + static_cast<size_t>(r) * step_src, static_cast<size_t>(w) * ch);
                }
            }
            ncs->mat = packed;
        }
        out->impl_ = std::make_shared<ImageDataImpl>();
        out->impl_->storage.reset(ncs);
        out->impl_->type = impl_->type;
        out->impl_->width = impl_->width;
        out->impl_->height = impl_->height;
        out->impl_->channels = impl_->channels;
        out->impl_->element_count_ = impl_->element_count_;
        out->impl_->element_bytes_ = impl_->element_bytes_;
        out->impl_->bytes_ = impl_->bytes_;
        return true;
    }

    bool ImageData::asMat(cv::Mat* out) const {
        g_last_error_msg.clear();
        if (!out) {
            g_last_error_msg = "asMat: null output";
            return false;
        }
        if (!impl_ || impl_->empty()) {
            g_last_error_msg = "asMat: image is empty";
            return false;
        }
        if (auto* cs = dynamic_cast<CpuStorage*>(impl_->storage.get())) {
            *out = cs->mat;
            return true;
        }
        if (auto* ps = dynamic_cast<PlaneStorage*>(impl_->storage.get())) {
            if (ps->device == Device::CPU && !ps->planes.empty() && ps->planes[0].data) {
                const int ocv_type = md_image_type_to_ocv_type(impl_->type);
                if (ocv_type > 0) {
                    // CPU 借用平面 → 包装成 mat（借用，不复制）
                    *out = cv::Mat(impl_->height, impl_->width, ocv_type,
                                   const_cast<uint8_t*>(ps->planes[0].data));
                    return true;
                }
            }
        }
        g_last_error_msg = "asMat: no CPU mat available to borrow";
        return false;
    }

    const char* ImageData::last_error() {
        return g_last_error_msg.empty() ? nullptr : g_last_error_msg.c_str();
    }


    ImageData& ImageData::rotate(const RotateFlags flag) {
        if (!impl_ || impl_->empty()) {
            return *this;
        }
        g_last_error_msg.clear();
        ImageData out;
        if (!backend_for(device())->rotate(*this, flag, &out)) {
            set_last_error("rotate: backend could not process the image (device frame or unsupported)");
            return *this;
        }
        *this = std::move(out);
        return *this;
    }


    ImageData ImageData::crop(const Rect2f& rect) const {
        if (!impl_ || impl_->empty()) {
            return ImageData();
        }
        g_last_error_msg.clear();
        ImageData out;
        if (backend_for(device())->crop(*this, rect.x, rect.y, rect.width, rect.height, &out)) {
            return out;
        }
        set_last_error("crop: backend could not process the image (device frame or unsupported)");
        return ImageData();
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
        impl_->mat()(roi & cv::Rect(0, 0, impl_->mat().cols, impl_->mat().rows)).copyTo(img_crop);
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
        g_last_error_msg.clear();
        ImageData out;
        if (backend_for(device())->resize(*this, &out, width, height)) {
            return out;
        }
        set_last_error("resize: backend could not process the image (device frame or unsupported)");
        return ImageData();
    }

    ImageData ImageData::cvt_color(const ImageData& image, const ColorConvertType type) {
        if (!image.impl_ || image.impl_->empty()) {
            return ImageData();
        }
        g_last_error_msg.clear();
        const auto ocv_type = md_color_convert_type_to_ocv_color_convert_type(type);
        if (ocv_type > 0) {
            // OpenCV 原生颜色转换：按设备经 backend 分派（设备帧未实现 → 报错，不静默）
            ImageData out;
            if (backend_for(image.device())->cvt_color(image, type, &out)) {
                return out;
            }
            set_last_error("cvt_color: backend could not process the image (device frame or unsupported)");
            return ImageData();
        }
        if (type == ColorConvertType::CVT_PA_BGR2PL_BGR || type == ColorConvertType::CVT_PA_RGB2PL_RGB) {
            // PL↔PA 拆合需要私有 impl 访问，留在本 TU 完成（与后端返回的 packed mat 等价）
            ImageData dst_image;
            dst_image.impl_ = std::make_shared<ImageDataImpl>();
            const int single_channel_type = CV_MAKETYPE(image.impl_->mat().depth(), 1);
            cv::Mat chw_image(image.channels(), image.height() * image.width(), single_channel_type);
            std::vector<cv::Mat> split_image;
            cv::split(image.impl_->mat(), split_image);
            for (int i = 0; i < split_image.size(); i++) {
                split_image[i] = split_image[i].reshape(1, 1);
                split_image[i].copyTo(chw_image.row(i));
            }
            auto* cs = new CpuStorage();
            cs->mat = chw_image.reshape(1, {image.channels(), image.height(), image.width()});
            dst_image.impl_->storage.reset(cs);
            dst_image.impl_->type = image.impl_->mat().depth() == CV_8U
                                        ? MdImageType::PLA_BGR_U8
                                        : MdImageType::PLA_BGR_F32;
            dst_image.impl_->refresh_meta();
            return dst_image;
        }
        if (type == ColorConvertType::CVT_PL_BGR2PA_BGR || type == ColorConvertType::CVT_PL_RGB2PA_RGB) {
            // valid chw format
            if (image.type() != MdImageType::PLA_BGR_U8 && image.type() != MdImageType::PLA_BGR_F32
                && image.type() != MdImageType::PLA_RGB_U8 && image.type() != MdImageType::PLA_RGB_F32) {
                set_last_error("cvt_color: invalid planar layout (expected Planar format)");
                return ImageData();
            }
            ImageData dst_image;
            dst_image.impl_ = std::make_shared<ImageDataImpl>();

            // 1 channel per row, total rows equal to channels
            cv::Mat planar_image = image.impl_->mat().reshape(1, image.channels());
            // 2. Split the planar image into separate channel matrices.
            std::vector<cv::Mat> split_images(image.channels());
            for (int i = 0; i < image.channels(); ++i) {
                split_images[i] = planar_image.row(i).reshape(1, image.height()); // reshape each row back to H x W
            }
            // 3. Merge these channel matrices into a single HWC image.
            cv::Mat hwc_image;
            cv::merge(split_images, hwc_image);

            auto* cs = new CpuStorage();
            cs->mat = hwc_image;
            dst_image.impl_->storage.reset(cs);
            dst_image.impl_->type = hwc_image.depth() == CV_8U
                                        ? MdImageType::PKG_BGR_U8
                                        : hwc_image.depth() == CV_32F
                                        ? MdImageType::PKG_BGR_F32
                                        : MdImageType::PKG_BGR_F64;
            dst_image.impl_->refresh_meta();
            return dst_image;
        }
        set_last_error("cvt_color: unsupported color conversion type");
        return ImageData();
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
            mat = impl_->mat().clone();
        }
        else {
            // Shallow copy: share underlying data
            mat = impl_->mat();
        }
    }


    std::vector<uint8_t> ImageData::imencode(const ImageData& image, const std::string& ext) {
        std::vector<uint8_t> buf;
        if (image.empty()) {
            MD_LOG_ERROR << "Cannot encode empty image" << std::endl;
            return buf;
        }
        cv::imencode(ext, image.impl_->mat(), buf);
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
        return cv::imwrite(filename, impl_->mat());
    }

    // 显示图片
    void ImageData::imshow(const std::string& win_name) const {
        if (!impl_ || impl_->empty()) {
            MD_LOG_ERROR << "Cannot display empty image" << std::endl;
            return;
        }
        cv::imshow(win_name, impl_->mat());
        cv::waitKey(0);
    }
}
