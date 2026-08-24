//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "vision/lpr/lpr_pipeline/lpr_pipeline.h"

namespace modeldeploy::vision {
    void bind_lpr_pipeline(const pybind11::module& m) {
        pybind11::class_<lpr::LprPipeline>(m, "LprPipeline")
            .def(pybind11::init([](const std::filesystem::path& det_model_path, const std::filesystem::path& rec_model_path, const RuntimeOption& option) {
                return std::make_unique<lpr::LprPipeline>(det_model_path.string(), rec_model_path.string(), option);
            }), pybind11::arg("det_model_path"), pybind11::arg("rec_model_path"))
            .def("predict", [](lpr::LprPipeline& self, const pybind11::array& image) {
                 const auto cv_image = pyarray_to_cv_mat(image);
                 std::vector<LprResult> results;
                 self.predict(ImageData(cv_image), &results);
                 return results;
             }, pybind11::arg("image"))
            .def("clone", [](const lpr::LprPipeline& self) {
                return self.clone();
            });
    }
}
