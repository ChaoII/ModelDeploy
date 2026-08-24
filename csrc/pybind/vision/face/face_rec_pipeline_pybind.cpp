//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_rec_pipeline/face_rec_pipeline.h"

namespace modeldeploy::vision {
    void bind_face_rec_pipeline(const pybind11::module& m) {
        pybind11::class_<face::FaceRecognizerPipeline, BaseModel>(m, "FaceRecognizerPipeline")
            .def(pybind11::init([](const std::filesystem::path& det_model_path, const std::filesystem::path& rec_model_path, const RuntimeOption& option) {
                return std::make_unique<face::FaceRecognizerPipeline>(det_model_path.string(), rec_model_path.string(), option);
            }), pybind11::arg("det_model_path"), pybind11::arg("rec_model_path"))
            .def("predict",
                  [](face::FaceRecognizerPipeline& self, pybind11::array& image) {
                      const auto mat = pyarray_to_cv_mat(image);
                      std::vector<FaceRecognitionResult> result;
                      self.predict(ImageData(mat), &result);
                      return result;
                  }, pybind11::arg("image"))
            .def("clone", [](const face::FaceRecognizerPipeline& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
