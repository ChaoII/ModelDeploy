//
// Created by aichao on 2025/7/18.
//

#include <array>
#include <cmath>
#include <algorithm>
#include "vision/utils.h"
#include "core/md_log.h"
#include "vision/common/convert.h"
#include "vision/common/image_data.h"
#include "vision/processors/processor_factory.h"
#include <opencv2/opencv.hpp>
#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif


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

    // 统一平面存储：唯一数据源 planes[]；owner 延续内存（自有缓冲 / 借用源 RAII）。
    // cv::Mat 仅作 asMat 结果的临时物化缓存，从不作为数据源。
    struct ImageDataImpl {
        MdImageType fmt = MdImageType::PKG_BGR_U8;
        int w = 0, h = 0, ch = 1;
        Device device = Device::CPU;
        std::array<ImageData::Plane, 3> planes{};
        size_t nplanes = 0;
        std::shared_ptr<void> owner;
        size_t bytes_ = 0, element_count_ = 0, element_bytes_ = 1;
    };

    static ImageDataImpl* get_impl(const ImageData& img) {
        return static_cast<ImageDataImpl*>(img.data_impl());
    }

    // 推导格式的通道数与 (w,h) 下总字节；返回 false 表示未知/不支持格式。
    static bool image_layout(const MdImageType fmt, const int w, const int h, int& ch, size_t& bytes) {
        const int ocv = md_image_type_to_ocv_type(fmt);
        if (ocv >= 0) {   // 含 GRAY_U8（CV_8UC1==0）与全部 packed 类型
            ch = CV_MAT_CN(ocv);
            bytes = static_cast<size_t>(w) * h * static_cast<size_t>(CV_ELEM_SIZE(ocv));
            return true;
        }
        if (fmt == MdImageType::NV12 || fmt == MdImageType::NV21 || fmt == MdImageType::I420) {
            ch = 1;
            bytes = static_cast<size_t>(w) * h * 3 / 2;
            return true;
        }
        // Planar (CHW) 平面布局
        if (is_planar_type(fmt)) {
            int elem = 1;
            switch (fmt) {
            case MdImageType::PLA_BGR_F32: case MdImageType::PLA_RGB_F32:
            case MdImageType::PLA_BGRA_F32: case MdImageType::PLA_RGBA_F32:
                elem = 4; break;
            default:
                break;
            }
            switch (fmt) {
            case MdImageType::PLA_BGRA_U8: case MdImageType::PLA_RGBA_U8:
            case MdImageType::PLA_BGRA_F32: case MdImageType::PLA_RGBA_F32:
                ch = 4; break;
            default:
                ch = 3; break;
            }
            bytes = static_cast<size_t>(w) * h * static_cast<size_t>(elem) * static_cast<size_t>(ch);
            return true;
        }
        return false;
    }

    // 构造自有内存的统一平面 ImageData（唯一数据源入口）。
    static ImageData make_owned(const MdImageType fmt, const int w, const int h, const Device device) {
        if (w <= 0 || h <= 0) return ImageData();
        int ch = 1;
        size_t bytes = 0;
        if (!image_layout(fmt, w, h, ch, bytes)) return ImageData();
        auto sp = std::make_shared<ImageDataImpl>();
        sp->fmt = fmt;
        sp->w = w;
        sp->h = h;
        sp->ch = ch;
        sp->device = device;
        sp->bytes_ = bytes;
        sp->element_count_ = static_cast<size_t>(w) * h;
        sp->element_bytes_ = bytes / (static_cast<size_t>(w) * h);
        auto buf = std::make_shared<std::vector<uint8_t>>(bytes);
        sp->owner = buf;
        uint8_t* p0 = buf->data();
        if (fmt == MdImageType::NV12 || fmt == MdImageType::NV21) {
            sp->planes[0] = {p0, w};
            sp->planes[1] = {p0 + static_cast<size_t>(w) * h, w};
            sp->nplanes = 2;
        } else if (fmt == MdImageType::I420) {
            sp->planes[0] = {p0, w};
            sp->planes[1] = {p0 + static_cast<size_t>(w) * h, w / 2};
            sp->planes[2] = {p0 + static_cast<size_t>(w) * h * 5 / 4, w / 2};
            sp->nplanes = 3;
        } else {
            sp->planes[0] = {p0, static_cast<int>(static_cast<size_t>(w) * ch)};
            sp->nplanes = 1;
        }
        ImageData img;
        img.take_impl(sp);
        return img;
    }

    // 逐平面搬运（尊重源步长，去处源数据填充），用于 clone / toCpu 深拷贝。
    // 设备(GPU)源用 cudaMemcpy2D 做 D2H；CPU 源用 std::memcpy。
    static void copy_planes_like(const ImageDataImpl* src, ImageDataImpl* dst) {
        const size_t n = (std::min)(src->nplanes, dst->nplanes);
        const bool src_on_gpu = (src->device == Device::GPU);
        for (size_t i = 0; i < n; ++i) {
            size_t rows = static_cast<size_t>(src->h);
            size_t stride = static_cast<size_t>(src->w) * static_cast<size_t>(src->ch);
            if (src->fmt == MdImageType::NV12 || src->fmt == MdImageType::NV21) {
                rows = (i == 0) ? static_cast<size_t>(src->h) : static_cast<size_t>(src->h) / 2;
                stride = static_cast<size_t>(src->w);
            } else if (src->fmt == MdImageType::I420) {
                rows = (i == 0) ? static_cast<size_t>(src->h) : static_cast<size_t>(src->h) / 2;
                stride = (i == 0) ? static_cast<size_t>(src->w) : static_cast<size_t>(src->w) / 2;
            }
            if (!src->planes[i].data || !dst->planes[i].data) continue;
            const uint8_t* s = src->planes[i].data;
            uint8_t* d = const_cast<uint8_t*>(dst->planes[i].data);
            const int step_src = src->planes[i].step > 0 ? src->planes[i].step : static_cast<int>(stride);
            const int step_dst = dst->planes[i].step;
            if (src_on_gpu) {
#ifdef WITH_GPU
                const cudaError_t err = cudaMemcpy2D(d, step_dst, s, step_src,
                                                     stride, rows, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    MD_LOG_WARN << "copy_planes_like: cudaMemcpy2D failed: "
                                << cudaGetErrorString(err) << std::endl;
                }
#else
                (void)rows; (void)stride; (void)step_src; (void)step_dst;
#endif
            } else {
                for (size_t r = 0; r < rows; ++r)
                    std::memcpy(d + static_cast<size_t>(r) * step_dst,
                                s + static_cast<size_t>(r) * step_src, stride);
            }
        }
    }

    ImageData::ImageData(const int width, const int height, const MdImageType type) {
        g_last_error_msg.clear();
        *this = make_owned(type, width, height, Device::CPU);
    }


    ImageData::ImageData(const cv::Mat& mat) {
        g_last_error_msg.clear();
        if (mat.empty()) {
            set_last_error("ImageData(Mat): empty mat");
            return;
        }
        const MdImageType t = md_image_type_from_ocv_type(mat.type());
        if (t == MdImageType::UNKNOWN) {
            set_last_error("ImageData(Mat): non-packed Mat type not supported");
            return;
        }
        *this = make_owned(t, mat.cols, mat.rows, Device::CPU);
        auto* d = get_impl(*this);
        if (d && d->nplanes == 1 && d->owner) {
            auto* buf = static_cast<std::vector<uint8_t>*>(d->owner.get());
            uint8_t* dst = buf->data();
            const size_t row_bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
            const size_t step_src = mat.step;
            for (int r = 0; r < mat.rows; ++r)
                std::memcpy(dst + static_cast<size_t>(r) * row_bytes,
                            mat.data + static_cast<size_t>(r) * step_src, row_bytes);
        }
    }

    ImageData ImageData::clone() const {
        g_last_error_msg.clear();
        if (!impl_ || empty()) return ImageData();
        auto* d = get_impl(*this);
        if (d->device != Device::CPU) {
            set_last_error("clone: device frame not supported");
            return ImageData();
        }
        ImageData img = make_owned(d->fmt, d->w, d->h, Device::CPU);
        if (img.empty()) {
            set_last_error("clone: unsupported format");
            return ImageData();
        }
        copy_planes_like(d, get_impl(img));
        return img;
    }


    ImageData ImageData::from_raw(unsigned char* data,
                                  const int width,
                                  const int height,
                                  const MdImageType type,
                                  const bool copy,
                                  const Device device,
                                  std::shared_ptr<void> owner) {
        g_last_error_msg.clear();
        if (!data || width <= 0 || height <= 0) {
            MD_LOG_ERROR << "Invalid parameters for from_raw" << std::endl;
            set_last_error("from_raw: invalid parameters");
            return ImageData();
        }
        if (copy) {
            ImageData img = make_owned(type, width, height, device);
            auto* d = get_impl(img);
            if (!d || !d->owner) {
                MD_LOG_ERROR << "Invalid MdImageType format: " << md_image_type_to_string(type) << std::endl;
                set_last_error("from_raw: unsupported format");
                return ImageData();
            }
            std::memcpy(static_cast<std::vector<uint8_t>*>(d->owner.get())->data(), data, d->bytes_);
            return img;
        }
        // 零拷贝借用
        int ch = 1;
        size_t bytes = 0;
        if (!image_layout(type, width, height, ch, bytes)) {
            MD_LOG_ERROR << "Invalid MdImageType format: " << md_image_type_to_string(type) << std::endl;
            set_last_error("from_raw: unsupported format");
            return ImageData();
        }
        auto sp = std::make_shared<ImageDataImpl>();
        sp->fmt = type;
        sp->w = width;
        sp->h = height;
        sp->ch = ch;
        sp->device = device;
        sp->bytes_ = bytes;
        sp->element_count_ = static_cast<size_t>(width) * height;
        sp->element_bytes_ = bytes / (static_cast<size_t>(width) * height);
        sp->owner = owner;
        uint8_t* p0 = const_cast<uint8_t*>(data);
        if (type == MdImageType::NV12 || type == MdImageType::NV21) {
            sp->planes[0] = {p0, width};
            sp->planes[1] = {p0 + static_cast<size_t>(width) * height, width};
            sp->nplanes = 2;
        } else if (type == MdImageType::I420) {
            sp->planes[0] = {p0, width};
            sp->planes[1] = {p0 + static_cast<size_t>(width) * height, width / 2};
            sp->planes[2] = {p0 + static_cast<size_t>(width) * height * 5 / 4, width / 2};
            sp->nplanes = 3;
        } else {
            sp->planes[0] = {p0, static_cast<int>(static_cast<size_t>(width) * ch)};
            sp->nplanes = 1;
        }
        ImageData img;
        img.take_impl(sp);
        return img;
    }


    int ImageData::width() const { return impl_ ? get_impl(*this)->w : 0; }
    int ImageData::height() const { return impl_ ? get_impl(*this)->h : 0; }
    int ImageData::channels() const { return impl_ ? get_impl(*this)->ch : 0; }
    MdImageType ImageData::type() const { return impl_ ? get_impl(*this)->fmt : MdImageType::PKG_BGR_U8; }
    MdImageType ImageData::format() const { return type(); }
    bool ImageData::empty() const {
        if (!impl_) return true;
        auto* d = get_impl(*this);
        return d->nplanes == 0 || !d->planes[0].data;
    }

    bool ImageData::is_shared_with(const ImageData& other) const {
        return impl_ && other.impl_ && impl_.get() == other.impl_.get();
    }

    size_t ImageData::element_count() const { return impl_ ? get_impl(*this)->element_count_ : 0; }
    size_t ImageData::element_bytes() const { return impl_ ? get_impl(*this)->element_bytes_ : 0; }
    size_t ImageData::bytes() const { return impl_ ? get_impl(*this)->bytes_ : 0; }

    Device ImageData::device() const { return impl_ ? get_impl(*this)->device : Device::CPU; }
    size_t ImageData::plane_count() const { return impl_ ? get_impl(*this)->nplanes : 0; }
    ImageData::Plane ImageData::plane(size_t i) const {
        if (!impl_) return {};
        auto* d = get_impl(*this);
        return i < d->nplanes ? d->planes[i] : Plane{};
    }

    ImageData ImageData::from_planes(const Plane* planes, const size_t n, const MdImageType fmt, const int w, const int h,
                                     const Device device, std::shared_ptr<void> owner) {
        g_last_error_msg.clear();
        if (!planes || n == 0 || n > 3 || w <= 0 || h <= 0) {
            set_last_error("from_planes: invalid parameters");
            return ImageData();
        }
        int ch = 1;
        size_t bytes = 0;
        if (!image_layout(fmt, w, h, ch, bytes)) {
            set_last_error("from_planes: unsupported format");
            return ImageData();
        }
        auto sp = std::make_shared<ImageDataImpl>();
        sp->fmt = fmt;
        sp->w = w;
        sp->h = h;
        sp->ch = ch;
        sp->device = device;
        sp->owner = owner;
        sp->bytes_ = bytes;
        sp->element_count_ = static_cast<size_t>(w) * h;
        sp->element_bytes_ = bytes / (static_cast<size_t>(w) * h);
        for (size_t i = 0; i < n; ++i) sp->planes[i] = planes[i];
        sp->nplanes = n;
        ImageData img;
        img.take_impl(sp);
        return img;
    }

    ImageData ImageData::from_bgr24(const uint8_t* bgr, int w, int h) {
        g_last_error_msg.clear();
        return from_raw(const_cast<unsigned char*>(bgr), w, h, MdImageType::PKG_BGR_U8, false, Device::CPU);
    }

    bool ImageData::toCpu(ImageData* out) const {
        g_last_error_msg.clear();
        if (!out) {
            set_last_error("toCpu: null output");
            return false;
        }
        if (!impl_ || empty()) {
            set_last_error("toCpu: source image is empty");
            return false;
        }
        auto* d = get_impl(*this);
        // NV21/I420 无正确 CPU 布局搬运实现 → 显式拒绝而非静默产出数据（NV12 保留平面搬运）
        if (d->fmt == MdImageType::NV21 || d->fmt == MdImageType::I420) {
            set_last_error("toCpu: unsupported YUV " + md_image_type_to_string(d->fmt));
            return false;
        }
        if (d->device == Device::CPU && d->nplanes == 1 && md_image_type_to_ocv_type(d->fmt) >= 0) {
            // 已是 CPU packed：浅 clone（共享 owner）
            auto sp = std::make_shared<ImageDataImpl>();
            sp->fmt = d->fmt;
            sp->w = d->w;
            sp->h = d->h;
            sp->ch = d->ch;
            sp->device = Device::CPU;
            sp->owner = d->owner;
            sp->planes = d->planes;
            sp->nplanes = d->nplanes;
            sp->bytes_ = d->bytes_;
            sp->element_count_ = d->element_count_;
            sp->element_bytes_ = d->element_bytes_;
            out->impl_ = sp;
            return true;
        }
        // 设备 / 多平面 → CPU 自有深拷贝（保持真实宽高与平面布局）
        ImageData img = make_owned(d->fmt, d->w, d->h, Device::CPU);
        if (img.empty()) {
            set_last_error("toCpu: unsupported format");
            return false;
        }
        copy_planes_like(d, get_impl(img));
        *out = std::move(img);
        return true;
    }

    bool ImageData::asMat(cv::Mat* out) const {
        g_last_error_msg.clear();
        if (!out) {
            set_last_error("asMat: null output");
            return false;
        }
        if (!impl_ || empty()) {
            set_last_error("asMat: image is empty");
            return false;
        }
        auto* d = get_impl(*this);
        if (d->device != Device::CPU || d->nplanes != 1 || !d->planes[0].data) {
            set_last_error("asMat: packed CPU only");
            return false;
        }
        const int ocv = md_image_type_to_ocv_type(d->fmt);
        if (ocv < 0) {
            set_last_error("asMat: packed CPU only");
            return false;
        }
        *out = cv::Mat(d->h, d->w, ocv, const_cast<uint8_t*>(d->planes[0].data), d->planes[0].step);
        return true;
    }

    std::vector<uint8_t> ImageData::to_native_bytes() const {
        ImageData cpu;
        if (device() != Device::CPU) {
            if (!toCpu(&cpu)) { set_last_error("to_native_bytes: 设备回读失败"); return {}; }
        } else {
            cpu = *this;
        }
        if (cpu.empty()) { set_last_error("to_native_bytes: 空图像"); return {}; }
        const int w = cpu.width(), h = cpu.height();
        const auto fmt = cpu.type();
        const size_t total = cpu.bytes();
        if (total == 0) { set_last_error("to_native_bytes: 零字节"); return {}; }

        auto plane_rows = [&](size_t i) -> int {
            if (fmt == MdImageType::NV12 || fmt == MdImageType::NV21) return i == 0 ? h : h / 2;
            if (fmt == MdImageType::I420) return i == 0 ? h : h / 2;
            return h;  // packed / planar / gray
        };

        std::vector<uint8_t> out(total);
        size_t off = 0;
        for (size_t i = 0; i < cpu.plane_count(); ++i) {
            const Plane p = cpu.plane(i);
            if (!p.data) continue;
            const int rows = plane_rows(i);
            const int step = p.step > 0 ? p.step : w;
            const size_t rowbytes = static_cast<size_t>(p.step > 0 ? p.step : w);
            for (int r = 0; r < rows && off < total; ++r) {
                const size_t n = (off + rowbytes <= total) ? rowbytes : (total - off);
                std::memcpy(out.data() + off, p.data + static_cast<size_t>(r) * step, n);
                off += n;
            }
        }
        return out;
    }

    const char* ImageData::last_error() {
        return g_last_error_msg.empty() ? nullptr : g_last_error_msg.c_str();
    }


    ImageData& ImageData::rotate(const RotateFlags flag) {
        if (!impl_ || empty()) {
            return *this;
        }
        g_last_error_msg.clear();
        if (!VisionProcessorBackend::supports(device(), ImageOp::Rotate)) {
            set_last_error("rotate: device unsupported (fast-fail)");
            // 不对称：fast-fail 清空本帧（返回类型 guard），而下方 backend 失败路径保留原位不动。
            ImageData empty;
            *this = std::move(empty);
            return *this;
        }
        ImageData out;
        if (!backend_for(device())->rotate(*this, flag, &out)) {
            set_last_error("rotate: backend could not process the image (device frame or unsupported)");
            return *this;
        }
        *this = std::move(out);
        return *this;
    }


    ImageData ImageData::crop(const Rect2f& rect) const {
        if (!impl_ || empty()) {
            return ImageData();
        }
        g_last_error_msg.clear();
        if (!VisionProcessorBackend::supports(device(), ImageOp::Crop)) {
            set_last_error("crop: device unsupported (fast-fail)");
            return ImageData();
        }
        ImageData out;
        if (backend_for(device())->crop(*this, rect.x, rect.y, rect.width, rect.height, &out)) {
            return out;
        }
        set_last_error("crop: backend could not process the image (device frame or unsupported)");
        return ImageData();
    }

    ImageData ImageData::rotate_crop(std::array<float, 8> box) const {
        if (!impl_ || empty()) {
            return ImageData();
        }
        g_last_error_msg.clear();
        if (!VisionProcessorBackend::supports(device(), ImageOp::RotateCrop)) {
            set_last_error("rotate_crop: device unsupported (fast-fail)");
            return ImageData();
        }
        ImageData out;
        if (backend_for(device())->rotate_crop(*this, box, &out)) {
            return out;
        }
        set_last_error("rotate_crop: backend could not process the image (device frame or unsupported)");
        return ImageData();
    }


    ImageData ImageData::resize(int width, int height) const {
        if (!impl_ || empty() || width <= 0 || height <= 0) {
            return ImageData();
        }
        g_last_error_msg.clear();
        if (!VisionProcessorBackend::supports(device(), ImageOp::Resize)) {
            set_last_error("resize: device unsupported (fast-fail)");
            return ImageData();
        }
        ImageData out;
        if (backend_for(device())->resize(*this, &out, width, height)) {
            return out;
        }
        set_last_error("resize: backend could not process the image (device frame or unsupported)");
        return ImageData();
    }

    ImageData ImageData::cvt_color(const ImageData& image, const ColorConvertType type) {
        if (!image.impl_ || image.empty()) {
            return ImageData();
        }
        g_last_error_msg.clear();
        if (!VisionProcessorBackend::supports(image.device(), ImageOp::CvtColor)) {
            set_last_error("cvt_color: device unsupported (fast-fail)");
            return ImageData();
        }
        // 对 CPU 帧：无正确实现的 YUV 转换显式拒绝，绝不静默产出垃圾。
        // NV21 全部无专用实现；I420 仅新增 CVT_I4202PKG_BGR 有专用实现。
        switch (type) {
        case ColorConvertType::CVT_NV212GRAY:
        case ColorConvertType::CVT_NV212PA_RGB:
        case ColorConvertType::CVT_NV212PA_BGR:
        case ColorConvertType::CVT_NV212PA_BGRA:
        case ColorConvertType::CVT_NV212PA_RGBA:
        case ColorConvertType::CVT_I4202GRAY:
        case ColorConvertType::CVT_I4202PA_BGR:
        case ColorConvertType::CVT_I4202PA_RGB:
        case ColorConvertType::CVT_I4202PA_BGRA:
        case ColorConvertType::CVT_I4202PA_RGBA:
            set_last_error("cvt_color: unsupported YUV conversion");
            return ImageData();
        default:
            break;
        }
        // 全部颜色转换（含 OpenCV 原生 + NV12/I420 + PL↔PA 拆合）统一经 backend 分派；
        // 设备帧未实现的 op 由 supports(CvtColor) 前置 fast-fail，不静默回退 CPU。
        ImageData out;
        if (backend_for(image.device())->cvt_color(image, type, &out)) {
            return out;
        }
        set_last_error("cvt_color: backend could not process the image (device frame or unsupported)");
        return ImageData();
    }

    // to_tensor/images_to_tensor 通用前置：统一平面存储下，只有"单平面 + 紧致"的图才能映射为
    // 单个 Tensor 平面（packed/GRAY）。多平面 YUV（NV12/I420）无法表达为单 tensor，显式拒绝而非静默丢 UV。
    // 返回 true 表示可通过 plane(0) 访问；否则置错返回 false。
    static bool tensor_mappable(const ImageData& img) {
        if (img.plane_count() != 1) {
            set_last_error("to_tensor: multi-plane YUV (NV12/I420) cannot map to a single tensor");
            return false;
        }
        // 紧致性：单平面 step（每行字节数）必须等于 w*ch，零拷贝共享/逐帧拷贝才不致错位。
        // 带填充（step>w*ch）的外部借用平面拒绝。
        if (img.plane(0).step != img.width() * img.channels()) {
            set_last_error("to_tensor: non-contiguous (padded) single plane cannot map to a tensor");
            return false;
        }
        return true;
    }

    void ImageData::images_to_tensor(const std::vector<ImageData>& images, Tensor* tensor) {
        if (images.empty() || !tensor) {
            set_last_error("images_to_tensor: empty images or null tensor");
            return;
        }
        g_last_error_msg.clear();
        const int n = static_cast<int>(images.size());
        const int c = images[0].channels();
        const int h = images[0].height();
        const int w = images[0].width();

        for (const auto& img : images) {
            if (img.channels() != c || img.width() != w || img.height() != h) {
                set_last_error("images_to_tensor: images shape is not equal");
                return;
            }
            // 批量是 CPU 侧"拷贝成 NCHW"语义：仅支持 CPU 单平面紧致；设备帧/多平面显式拒绝
            // （设备零拷贝走 predict/from_planes 链路，不在此处做隐含 D2H）。
            if (img.device() != Device::CPU || !tensor_mappable(img)) {
                set_last_error("images_to_tensor: only CPU single-plane contiguous images supported");
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
            const uint8_t* src = images[i].plane(0).data;
            if (src) {
                std::memcpy(base + i * bytes, src, bytes);
            }
        }
    }

    void ImageData::to_tensor(Tensor* tensor, const bool copy) {
        if (!impl_ || empty()) {
            set_last_error("to_tensor: image empty");
            return;
        }
        if (!tensor) {
            set_last_error("to_tensor: null tensor");
            return;
        }
        g_last_error_msg.clear();
        // 统一平面存储 + 设备感知：先把不可映射/跨设备情况显式拦下，杜绝"默认 CPU"的隐式拷贝。
        if (!tensor_mappable(*this)) {
            return;
        }
        const auto dtype = utils::md_image_dtype_to_md_dtype(type());
        const std::vector<int64_t> shape = {channels(), height(), width()};
        if (copy) {
            // copy=true 产生 CPU 副本（Tensor::allocate 仅分配 CPU 内存）。
            // 设备帧的"拷贝"本身就需要一次 D2H —— 与"全程零拷贝、无 D2H/H2D"的主线相悖，显式拒绝。
            if (device() != Device::CPU) {
                set_last_error("to_tensor: copy=true not supported for device frame; use copy=false zero-copy");
                return;
            }
            const size_t num_bytes = bytes();
            // allocate 复用逻辑：shape/dtype/device 不变时复用已有 MemoryBlock
            tensor->allocate(shape, dtype);
            if (num_bytes != tensor->byte_size()) {
                set_last_error("to_tensor: tensor size mismatch, tensor=" +
                               std::to_string(tensor->byte_size()) + ", image=" + std::to_string(num_bytes));
                return;
            }
            if (plane(0).data && tensor->data()) {
                memcpy(tensor->data(), plane(0).data, num_bytes);
            }
        }
        else {
            // 零拷贝：把 plane(0) 按该内存所属 device（CPU/GPU/TPU）包装成 Tensor，共享外部内存，不做任何拷贝。
            // 设备帧的 plane(0).data 指向设备内存，故产出设备 Tensor，绝无 H2D/D2H。
            tensor->from_external_memory(const_cast<uint8_t*>(plane(0).data), shape, dtype, nullptr, device());
        }
    }


    // 编解码唯一支持的图像：CPU + 单平面（packed/gray）+ OpenCV 可映射格式。
    // 设备帧 / NV12 等多平面帧 → false（不静默回退 CPU、不操作空 mat，避免 OpenCV 断言）。
    static bool codec_supported(const ImageData& image) {
        if (image.device() != Device::CPU) return false;
        if (image.plane_count() > 1) return false;
        // 阈值为 >= 0（非 > 0）：刻意允许 GRAY_U8（CV_8UC1==0）参与编解码 —— OpenCV 能正确编码灰度；
        // 这是修复过去错误拒绝灰度图的意图，勿当作意外的放宽。
        return md_image_type_to_ocv_type(image.format()) >= 0;
    }

    std::vector<uint8_t> ImageData::imencode(const ImageData& image, const std::string& ext) {
        g_last_error_msg.clear();
        std::vector<uint8_t> buf;
        if (image.empty()) {
            g_last_error_msg = "imencode: image is empty";
            return buf;
        }
        if (!codec_supported(image)) {
            g_last_error_msg = "imencode: only CPU single-plane image supported";
            return buf;
        }
        if (image.format() == MdImageType::NV21 || image.format() == MdImageType::I420) {
            g_last_error_msg = "imencode: YUV NV21/I420 encoding unsupported (CPU single-plane only)";
            return buf;
        }
        cv::Mat m;
        if (!image.asMat(&m)) {
            g_last_error_msg = "imencode: cannot map image to Mat";
            return buf;
        }
        cv::imencode(ext, m, buf);
        return buf;
    }

    ImageData ImageData::imdecode(const std::vector<uint8_t>& buf) {
        cv::Mat mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
        if (mat.empty()) return ImageData();
        return ImageData(mat);
    }


    ImageData ImageData::imread(const std::string& filename) {
        cv::Mat mat = cv::imread(filename);
        if (mat.empty()) {
            MD_LOG_ERROR << "Failed to read image: " << filename << std::endl;
            return ImageData();
        }
        return ImageData(mat);
    }

    bool ImageData::imwrite(const std::string& filename) const {
        g_last_error_msg.clear();
        if (!impl_ || empty()) {
            g_last_error_msg = "imwrite: image is empty";
            return false;
        }
        if (!codec_supported(*this)) {
            g_last_error_msg = "imwrite: only CPU single-plane image supported";
            return false;
        }
        if (format() == MdImageType::NV21 || format() == MdImageType::I420) {
            g_last_error_msg = "imwrite: YUV NV21/I420 encoding unsupported (CPU single-plane only)";
            return false;
        }
        cv::Mat m;
        if (!asMat(&m)) {
            g_last_error_msg = "imwrite: cannot map image to Mat";
            return false;
        }
        return cv::imwrite(filename, m);
    }

    // 显示图片
    void ImageData::imshow(const std::string& win_name) const {
        if (!impl_ || empty()) {
            MD_LOG_ERROR << "Cannot display empty image" << std::endl;
            return;
        }
        g_last_error_msg.clear();
        if (device() != Device::CPU) {
            set_last_error("imshow: CPU only");
            return;
        }
        cv::Mat m;
        if (!asMat(&m)) {
            set_last_error("imshow: CPU packed image only");
            return;
        }
        cv::imshow(win_name, m);
        cv::waitKey(0);
    }
}
