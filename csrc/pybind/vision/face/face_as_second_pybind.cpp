//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_as/face_as_second.h"

namespace modeldeploy::vision {
    void bind_face_as_second(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsSecond, BaseModel>(m, "SeetaFaceAsSecond")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<face::SeetaFaceAsSecond>(model_file.string(), option);
            }), pybind11::arg("model_file"))
            .def("predict",
                  [](face::SeetaFaceAsSecond& self, pybind11::array& image) {
                      auto mat = pyarray_to_cv_mat(image);
                      std::vector<std::tuple<int, float>> result;
                      self.predict(ImageData(mat), &result);
                      return result;
                  }, pybind11::arg("image"))
            .def("clone", [](const face::SeetaFaceAsSecond& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
