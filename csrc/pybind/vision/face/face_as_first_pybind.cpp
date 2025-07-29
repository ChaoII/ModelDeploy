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
                 [](face::SeetaFaceAsFirst& self, const pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     const auto image_data = ImageData::from_mat(&mat);
                     float result;
                     self.predict(image_data, &result);
                     return result;
                 }, pybind11::arg("image"));
    }
} // namespace modeldeploy
