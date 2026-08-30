//
// Created by aichao on 2025/8/30.
//

#include "pybind/utils/utils.h"
#include "vision/sam/fastsam.h"

namespace modeldeploy::vision {
    void bind_fastsam(const pybind11::module& m) {
        pybind11::class_<seg::FastSamPreprocessor>(m, "FastSamPreprocessor")
            .def(pybind11::init<>());

        pybind11::class_<seg::FastSamPostprocessor>(m, "FastSamPostprocessor")
            .def(pybind11::init<>());

        pybind11::class_<seg::FastSam, BaseModel>(m, "FastSam")
            .def(pybind11::init([](const std::filesystem::path& model_file, const RuntimeOption& option) {
                return std::make_unique<seg::FastSam>(model_file.string(), option);
            }), pybind11::arg("model_file"), pybind11::arg("option"))
            .def("predict",
                 [](seg::FastSam& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::vector<InstanceSegResult> result;
                     self.predict(ImageData(mat), &result);
                     return result;
                 }, pybind11::arg("image"))
            .def("predict_with_prompts",
                 [](seg::FastSam& self, pybind11::array& image,
                    const std::vector<std::vector<float>>& bboxes,
                    const std::vector<std::vector<float>>& points,
                    const std::vector<int>& point_labels) {
                     const auto mat = pyarray_to_cv_mat(image);
                     seg::FastSamPrompts prompts;
                     for (const auto& b : bboxes) {
                         if (b.size() >= 4) {
                             prompts.bboxes.emplace_back(Rect2f(b[0], b[1], b[2], b[3]));
                         }
                     }
                     for (const auto& p : points) {
                         if (p.size() >= 2) {
                             prompts.points.emplace_back(Point2f(p[0], p[1]));
                         }
                     }
                     prompts.point_labels = point_labels;
                     std::vector<InstanceSegResult> result;
                     self.predict_with_prompts(ImageData(mat), prompts, &result);
                     return result;
                 }, pybind11::arg("image"), pybind11::arg("bboxes"),
                    pybind11::arg("points"), pybind11::arg("point_labels"))
            .def("batch_predict",
                 [](seg::FastSam& self, const std::vector<pybind11::array>& images) {
                     std::vector<ImageData> _images;
                     _images.reserve(images.size());
                     for (auto& image : images) {
                         auto cv_image = pyarray_to_cv_mat(image);
                         _images.push_back(ImageData(cv_image));
                     }
                     std::vector<std::vector<InstanceSegResult>> results;
                     self.batch_predict(_images, &results);
                     return results;
                 }, pybind11::arg("images"))
            .def_property_readonly("preprocessor", &seg::FastSam::get_preprocessor)
            .def_property_readonly("postprocessor", &seg::FastSam::get_postprocessor)
            .def("clone", [](const seg::FastSam& self) {
                return self.clone();
            });
    }
} // namespace modeldeploy::vision
