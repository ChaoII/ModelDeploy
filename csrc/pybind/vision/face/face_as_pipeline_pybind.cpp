//
// Created by aichao on 2025/6/10.
//


#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/face/face_as/face_as_pipeline.h"

namespace modeldeploy::vision {
    void bind_as_pipeline(const pybind11::module& m) {
        pybind11::class_<face::SeetaFaceAsPipeline, BaseModel>(m, "SeetaFaceAsPipeline")
            .def(pybind11::init<std::string, std::string, std::string, RuntimeOption>())
            .def("predict",
                 [](const face::SeetaFaceAsPipeline& self, pybind11::array& data, const float fuse_threshold,
                    const float clarity_threshold) {
                     const auto mat = pyarray_to_cv_mat(data);
                     std::vector<FaceAntiSpoofResult> results;
                     self.predict(mat, &results, fuse_threshold, clarity_threshold);
                     return results;
                 });
    }
} // namespace modeldeploy
