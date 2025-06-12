//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h"

namespace modeldeploy::vision {
    void bind_lpr_pipeline(const pybind11::module& m) {
        pybind11::class_<lpr::LprPipeline>(m, "LprPipeline")
            .def(pybind11::init<std::string, std::string, RuntimeOption>())
            .def("predict", [](lpr::LprPipeline& self, pybind11::array& image) {
                std::vector<LprResult> results;
                self.predict(pyarray_to_cv_mat(image), &results);
                return results;
            }, pybind11::arg("image"));
    }
}
