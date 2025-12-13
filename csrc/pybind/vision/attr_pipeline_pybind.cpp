//
// Created by aichao on 2025/12/13.
//

#include "pybind/utils/utils.h"
#include "vision/pipeline/pedestrian_attribute.h"

namespace modeldeploy::vision {
    void bind_attr_pipeline(const pybind11::module& m) {
        pybind11::class_<pipeline::PedestrianAttribute, BaseModel>(m, "PedestrianAttribute")
            .def(pybind11::init<std::string, std::string, RuntimeOption>(),
                 pybind11::arg("det_model_path"),
                 pybind11::arg("cls_model_path"),
                 pybind11::arg("option"))
            .def("predict",
                 [](pipeline::PedestrianAttribute& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<AttributeResult> result;
                     self.predict(ImageData::from_mat(&mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("batch_predict",
                 [](pipeline::PedestrianAttribute& self, std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData::from_mat(&cv_image));
                     }
                     std::vector<std::vector<AttributeResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property("cls_batch_size",
                          &pipeline::PedestrianAttribute::get_cls_batch_size,
                          &pipeline::PedestrianAttribute::set_cls_batch_size)
            .def_property("det_input_size",
                          &pipeline::PedestrianAttribute::get_det_input_size,
                          &pipeline::PedestrianAttribute::set_det_input_size)
            .def_property("cls_input_size",
                          &pipeline::PedestrianAttribute::get_cls_input_size,
                          &pipeline::PedestrianAttribute::set_cls_input_size)
            .def("set_det_threshold", &pipeline::PedestrianAttribute::set_det_threshold)
            .def("get_detector", &pipeline::PedestrianAttribute::get_detector)
            .def("get_classifier", &pipeline::PedestrianAttribute::get_classifier);
    }
} // namespace modeldeploy
