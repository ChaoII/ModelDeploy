#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind/utils/utils.h"
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"
#include "vision/common/result.h"

namespace modeldeploy::vision {
    void bind_reid(const pybind11::module& m) {
        pybind11::class_<ReIdResult>(m, "ReIdResult")
            .def(pybind11::init<>())
            .def_readwrite("embedding", &ReIdResult::embedding);

        pybind11::class_<reid::ReID>(m, "ReID")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<reid::ReID>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](reid::ReID& self, const pybind11::array& im) {
                     const auto cv = pyarray_to_cv_mat(im);
                     std::vector<ReIdResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out))
                         throw std::runtime_error("ReID predict failed");
                     return out;
                 },
                 pybind11::arg("image"));

        pybind11::class_<reid::ReIdGallery>(m, "ReIdGallery")
            .def(pybind11::init<>())
            .def("enroll", &reid::ReIdGallery::enroll,
                 pybind11::arg("label"), pybind11::arg("embedding"))
            .def("match", &reid::ReIdGallery::match,
                 pybind11::arg("embedding"), pybind11::arg("k"))
            .def("remove", &reid::ReIdGallery::remove, pybind11::arg("label"))
            .def("clear", &reid::ReIdGallery::clear)
            .def("size", &reid::ReIdGallery::size);
    }
}
