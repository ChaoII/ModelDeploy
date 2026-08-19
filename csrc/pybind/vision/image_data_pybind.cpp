//
// Python 侧 ImageData 绑定：外部构造 NV12（host/device）帧，统一零拷贝 NV12 推理路径。
//  - from_nv12         : host NV12，库内拷入自有缓冲（安全，无需调用方保活）——与 capi2 md_image_from_nv12_owned 一致。
//  - from_device_nv12  : 设备/主机 NV12 零拷贝借用（y/uv 指向调用方内存，不拷贝）；调用方须保证缓冲在 predict 期间存活。
// 需模型文件（[model] 标签，CI 下载）时才走真实推理；本绑定仅为构造帧，供 predict/ImageData 路径使用。
//

#include "pybind/utils/utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include <cstring>
#include "vision/common/image_data.h"

namespace modeldeploy::vision {

    namespace {
        // 便捷：把不从属的 Plane 指针 + 可选 owner 组装成 ImageData（host CPU NV12）
        ImageData make_cpu_nv12(const uint8_t* y, const uint8_t* uv, int w, int h,
                                int step_y, int step_uv, std::shared_ptr<void> owner) {
            if (w <= 0 || h <= 0) {
                return ImageData();
            }
            if (step_y <= 0) step_y = w;
            if (step_uv <= 0) step_uv = w;
            ImageData::Plane pl[2] = {
                {y, step_y},
                {uv, step_uv},
            };
            return ImageData::from_planes(pl, uv ? 2 : 1, MdImageType::NV12, w, h, Device::CPU, std::move(owner));
        }
    }

    void bind_image_data(const pybind11::module& m) {
        pybind11::class_<ImageData>(m, "ImageData")
            .def(pybind11::init<>())
            .def_property_readonly("width", [](const ImageData& im) { return im.width(); })
            .def_property_readonly("height", [](const ImageData& im) { return im.height(); })
            .def_property_readonly("type", [](const ImageData& im) { return static_cast<int>(im.type()); })
            .def("empty", [](const ImageData& im) { return im.empty(); })
            .def("plane_count", [](const ImageData& im) { return im.plane_count(); })

            // 自有（安全）host NV12：y/uv 为 uint8 数组；step 为每行字节步长，0=紧凑。
            .def_static(
                "from_nv12",
                [](const pybind11::array& y, const pybind11::array& uv,
                   int w, int h, int step_y, int step_uv) -> ImageData {
                    if (y.ndim() != 1 || !(y.dtype().is(pybind11::dtype::of<uint8_t>()))) {
                        throw std::runtime_error("from_nv12: y must be a 1-D uint8 numpy array");
                    }
                    if (step_y <= 0) step_y = w;
                    const size_t y_bytes = static_cast<size_t>(step_y) * static_cast<size_t>(h);
                    const size_t uv_bytes = (uv.is_none() || uv.ndim() == 0)
                        ? 0
                        : static_cast<size_t>(step_uv > 0 ? step_uv : w) * (static_cast<size_t>(h) / 2);
                    void* ybuf = y.request().ptr;
                    const uint8_t* uvptr = (uv.is_none() || uv.ndim() == 0) ? nullptr
                        : static_cast<const uint8_t*>(uv.request().ptr);

                    auto* mem = new uint8_t[y_bytes + (uvptr ? uv_bytes : 0)];
                    std::memcpy(mem, ybuf, y_bytes);
                    uint8_t* yown = mem;
                    uint8_t* uvown = mem + y_bytes;
                    if (uvptr) std::memcpy(uvown, uvptr, uv_bytes);
                    auto owner = std::shared_ptr<void>(mem, [](void* p) { delete[] static_cast<uint8_t*>(p); });

                    return make_cpu_nv12(yown, uvown, w, h, step_y, uvptr ? step_uv : 0, std::move(owner));
                },
                pybind11::arg("y"), pybind11::arg("uv"), pybind11::arg("width"), pybind11::arg("height"),
                pybind11::arg("step_y") = 0, pybind11::arg("step_uv") = 0)

            // 零拷贝借用（host/device）：y/uv 指向调用方内存，不拷贝；调用方须保活。
            .def_static(
                "from_device_nv12",
                [](const pybind11::array& y, const pybind11::array& uv,
                   int w, int h, int step_y, int step_uv) -> ImageData {
                    if (y.ndim() != 1 || !(y.dtype().is(pybind11::dtype::of<uint8_t>()))) {
                        throw std::runtime_error("from_device_nv12: y must be a 1-D uint8 numpy array");
                    }
                    const uint8_t* yptr = static_cast<const uint8_t*>(y.request().ptr);
                    const uint8_t* uvptr = uv.is_none() || uv.ndim() == 0
                        ? nullptr : static_cast<const uint8_t*>(uv.request().ptr);
                    return make_cpu_nv12(yptr, uvptr, w, h, step_y, step_uv, {});
                },
                pybind11::arg("y"), pybind11::arg("uv"), pybind11::arg("width"), pybind11::arg("height"),
                pybind11::arg("step_y") = 0, pybind11::arg("step_uv") = 0)

            .def("__repr__", [](const ImageData& im) {
                return "ImageData(width=" + std::to_string(im.width()) +
                       ", height=" + std::to_string(im.height()) +
                       ", type=" + std::to_string(static_cast<int>(im.type())) +
                       ", empty=" + (im.empty() ? "true" : "false") + ")";
            });
    }

} // namespace modeldeploy::vision
