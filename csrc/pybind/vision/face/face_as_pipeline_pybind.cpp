//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_as/face_as_pipeline.h"

namespace modeldeploy::vision {
    void bind_as_pipeline(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsPipeline, BaseModel>(m, "SeetaFaceAsPipeline")
            .def(pybind11::init([](const std::filesystem::path& face_det_model_file, const std::filesystem::path& first_model_file, const std::filesystem::path& second_model_file, const RuntimeOption& option) {
                return std::make_unique<face::SeetaFaceAsPipeline>(face_det_model_file.string(), first_model_file.string(), second_model_file.string(), option);
            }), pybind11::arg("face_det_model_file"), pybind11::arg("first_model_file"), pybind11::arg("second_model_file"), pybind11::arg("option"))
            .def("predict",
                  [](const face::SeetaFaceAsPipeline& self, pybind11::array& image, const float fuse_threshold,
                     const float clarity_threshold) {
                      const auto mat = pyarray_to_cv_mat(image);
                      std::vector<FaceAntiSpoofResult> results;
                      self.predict(ImageData(mat), &results, fuse_threshold, clarity_threshold);
                      return results;
                  }, pybind11::arg("image"),
                  pybind11::arg("fuse_threshold") = 0.8,
                  pybind11::arg("clarity_threshold") = 0.3)
            .def("clone", [](const face::SeetaFaceAsPipeline& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
