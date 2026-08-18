//
// Created by aichao on 2026/8/13.
//

#include "pybind/utils/utils.h"
#include "vision/sem/ultralytics_sem.h"

namespace modeldeploy::vision {
    void bind_ultralytics_sem(const pybind11::module& m) {
        pybind11::class_<detection::UltralyticsSemPreprocessor>(m, "UltralyticsSemPreprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsSemPreprocessor& self,
                    std::vector<pybind11::array>& im_list) {
                     std::vector<ImageData> images;
                     images.reserve(im_list.size());
                     for (auto& image : im_list) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         images.push_back(ImageData(cv_image));
                     }
                     std::vector<LetterBoxRecord> records;
                     std::vector<Tensor> outputs;
                     if (!self.run(images, &outputs, &records)) {
                         throw std::runtime_error(
                             "Failed to preprocess the input data in UltralyticsSemPreprocessor.");
                     }
                     return make_pair(std::move(outputs), std::move(records));
                 }, pybind11::arg("im_list"))
            .def_property("size", &detection::UltralyticsSemPreprocessor::get_size,
                          &detection::UltralyticsSemPreprocessor::set_size)
            .def_property("padding_value",
                          &detection::UltralyticsSemPreprocessor::get_padding_value,
                          &detection::UltralyticsSemPreprocessor::set_padding_value);

        pybind11::class_<detection::UltralyticsSemPostprocessor>(
                m, "UltralyticsSemPostprocessor")
            .def(pybind11::init<>())
            .def("run",
                 [](const detection::UltralyticsSemPostprocessor& self,
                    const std::vector<Tensor>& inputs,
                    const std::vector<LetterBoxRecord>& records) {
                     std::vector<SemSegResult> results;
                     if (!self.run(inputs, &results, records)) {
                         throw std::runtime_error(
                             "Failed to postprocess the runtime result in "
                             "UltralyticsSemPostprocessor.");
                     }
                     return results;
                 }, pybind11::arg("inputs"), pybind11::arg("records"));

        pybind11::class_<detection::UltralyticsSem, BaseModel>(m, "UltralyticsSem")
            .def(pybind11::init<std::string, RuntimeOption>())
            .def("predict",
                 [](detection::UltralyticsSem& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     SemSegResult result;
                     self.predict(ImageData(mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](detection::UltralyticsSem& self,
                    std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData(cv_image));
                     }
                     std::vector<SemSegResult> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor",
                                   &detection::UltralyticsSem::get_preprocessor)
            .def_property_readonly("postprocessor",
                                    &detection::UltralyticsSem::get_postprocessor)
            .def("clone", [](const detection::UltralyticsSem& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy
