//
// Created by aichao on 2025/6/10.
//


#include "pybind/utils/utils.h"
#include "vision/face/face_as/face_as_pipeline.h"

namespace modeldeploy::vision {
    void bind_as_pipeline(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsPipeline, BaseModel>(m, "SeetaFaceAsPipeline")
            .def(pybind11::init<std::string, std::string, std::string, RuntimeOption>())
            .def("predict",
                 [](const face::SeetaFaceAsPipeline& self, pybind11::array& image, const float fuse_threshold,
                    const float clarity_threshold) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<FaceAntiSpoofResult> results;
                     self.predict(ImageData::from_mat(&mat), &results, fuse_threshold, clarity_threshold);
                     return results;
                 }, pybind11::arg("image"),
                 pybind11::arg("fuse_threshold") = 0.8,
                 pybind11::arg("clarity_threshold") = 0.3);
    }
} // namespace modeldeploy
