//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_rec_pipeline/face_rec_pipeline.h"

namespace modeldeploy::vision {
    void bind_face_rec_pipeline(const pybind11::module& m) {
        pybind11::class_<face::FaceRecognizerPipeline, BaseModel>(m, "FaceRecognizerPipeline")
            .def(pybind11::init<std::string, std::string, RuntimeOption>())
            .def("predict",
                 [](face::FaceRecognizerPipeline& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<FaceRecognitionResult> result;
                     self.predict(mat, &result);
                     return result;
                 }, pybind11::arg("image"));
    }
} // namespace modeldeploy
