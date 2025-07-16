//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_as/face_as_first.h"

namespace modeldeploy::vision {
    void bind_face_as_first(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsFirst, BaseModel>(m, "SeetaFaceAsFirst")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceAsFirst& self, pybind11::array& image) {
                     auto mat = pyarray_to_cv_mat(image);
                     float result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"));
    }
} // namespace modeldeploy
