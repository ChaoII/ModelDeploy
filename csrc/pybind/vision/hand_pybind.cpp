//
// Created by aichao on 2026/08/22.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind/utils/utils.h"
#include "vision/hand/hand.h"

namespace modeldeploy::vision {
    void bind_hand(const pybind11::module& m) {
        pybind11::class_<hand::HandKeypoint>(m, "HandKeypoint")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<hand::HandKeypoint>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](hand::HandKeypoint& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out)) {
                         throw std::runtime_error("HandKeypoint predict failed");
                     }
                     return out;
                 }, pybind11::arg("image"))
            .def("set_keypoints_num",
                 [](hand::HandKeypoint& self, int n) {
                     self.get_postprocessor().set_keypoints_num(n);
                 }, pybind11::arg("n"))
            .def("get_keypoints_num",
                 [](hand::HandKeypoint& self) {
                     return self.get_postprocessor().get_keypoints_num();
                 });
    }
} // namespace modeldeploy::vision
