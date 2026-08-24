//
// Python 侧 VideoDecoder 绑定（Task 7）：modeldeploy.video.VideoDecoder。
// next() 复用 vision 子模块已注册的 ImageData；仅 BUILD_VIDEO&&BUILD_VISION 时注册。
//

#include <pybind11/pybind11.h>
#include <utility>

#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
#include "csrc/video/video_decoder.h"
#include "pybind/utils/utils.h"

namespace py = pybind11;

namespace modeldeploy::video {

void bind_video(pybind11::module& m) {
    py::class_<VideoDecoder>(m, "VideoDecoder")
        .def(py::init<>())
        .def("open", [](VideoDecoder& d, const std::filesystem::path& url) {
            return d.open(url.string());
        }, pybind11::arg("url"))
        .def("next", [](VideoDecoder& d) {
            modeldeploy::vision::ImageData f;
            uint64_t pts = 0;
            bool ok = d.next(&f, &pts);
            return std::make_pair(ok, f);
        })
        .def("close", &VideoDecoder::close)
        .def_property_readonly("width", &VideoDecoder::width)
        .def_property_readonly("height", &VideoDecoder::height)
        .def_property_readonly("fps", &VideoDecoder::fps);
}

} // namespace modeldeploy::video
#endif
