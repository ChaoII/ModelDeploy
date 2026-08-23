//
// Created by aichao on 2026/08/23.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind/utils/utils.h"
#include "vision/landmark/face_landmark.h"
#include "vision/landmark/vehicle_keypoint.h"

namespace modeldeploy::vision {
    void bind_landmark(pybind11::module& m) {
        // modeldeploy.vision.landmark：车辆关键点（VehicleKeypoint / 复用 KeyPointsResult）。
        auto landmark_m = m.def_submodule(
            "landmark",
            "Landmark / keypoint models of Modeldeploy: VehicleKeypoint (vehicle wheels) and "
            "FaceLandmark (InsightFace 2d106 facial landmarks).");

        pybind11::class_<landmark::VehicleKeypoint>(landmark_m, "VehicleKeypoint",
                                                    "车辆关键点识别（默认 4 车轮关键点，薄封装 UltralyticsPose，"
                                                    "可 set_keypoints_num 泛化点数）。")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption(),
                 "构造 VehicleKeypoint；无权重时 is_initialized()==False。")
            .def("predict",
                 [](landmark::VehicleKeypoint& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out)) {
                         throw std::runtime_error("VehicleKeypoint predict failed");
                     }
                     return out;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](landmark::VehicleKeypoint& self, const std::vector<pybind11::array>& images) {
                     std::vector<ImageData> imgs;
                     imgs.reserve(images.size());
                     for (auto& image : images) {
                         auto cv = pyarray_to_cv_mat(image);
                         imgs.push_back(ImageData(cv));
                     }
                     std::vector<std::vector<KeyPointsResult>> out;
                     if (!self.batch_predict(imgs, &out)) {
                         throw std::runtime_error("VehicleKeypoint batch_predict failed");
                     }
                     return out;
                 }, pybind11::arg("images"))
            .def("set_keypoints_num",
                 [](landmark::VehicleKeypoint& self, int n) {
                     self.get_postprocessor().set_keypoints_num(n);
                 }, pybind11::arg("n"))
            .def("get_keypoints_num",
                 [](landmark::VehicleKeypoint& self) {
                     return self.get_postprocessor().get_keypoints_num();
                 })
            .def("is_initialized", &landmark::VehicleKeypoint::is_initialized)
            .def("clone", [](const landmark::VehicleKeypoint& self) {
                return self.clone();
            });

        pybind11::class_<landmark::FaceLandmark>(landmark_m, "FaceLandmark",
                                                 "面部 Landmark 独立访问（InsightFace 2d106，106 点）。")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption(),
                 "构造 FaceLandmark；无权重时 is_initialized()==False。")
            .def("predict",
                 [](landmark::FaceLandmark& self, const pybind11::array& im) {
                     auto cv = pyarray_to_cv_mat(im);
                     std::vector<KeyPointsResult> out;
                     ImageData img(cv);
                     if (!self.predict(img, &out)) {
                         throw std::runtime_error("FaceLandmark predict failed");
                     }
                     return out;
                 }, pybind11::arg("image"))
            .def("is_initialized", &landmark::FaceLandmark::is_initialized)
            .def("clone", [](const landmark::FaceLandmark& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy::vision
