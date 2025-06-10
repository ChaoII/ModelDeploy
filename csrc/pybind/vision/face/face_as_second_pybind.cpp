//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_as/face_as_second.h"

namespace modeldeploy::vision {
    void bind_face_as_second(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsSecond, BaseModel>(m, "SeetaFaceAsSecond")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](face::SeetaFaceAsSecond& self, pybind11::array& data) {
                     auto mat = pyarray_to_cv_mat(data);
                     std::vector<std::tuple<int, float>> res;
                     self.predict(mat, &res);
                     return res;
                 });
    }
} // namespace modeldeploy
