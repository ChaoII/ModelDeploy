//
// Created by aichao on 2026/8/23.
//

#include "pybind/utils/utils.h"
#include "vision/ocr/formula_recognition.h"
#include "vision/ocr/doc_to_markdown.h"

namespace modeldeploy::vision {
    void bind_formula_recognizer(const pybind11::module& m) {
        // FormulaRecognizer: cropped formula image -> LaTeX string
        pybind11::class_<ocr::FormulaRecognizer, BaseModel>(m, "FormulaRecognizer")
            .def(pybind11::init([](const std::filesystem::path& model_file, const std::filesystem::path& char_dict_path, pybind11::object option_obj) {
                RuntimeOption option = pybind11::none().equal(option_obj) ? RuntimeOption() : option_obj.cast<RuntimeOption>();
                return std::make_unique<ocr::FormulaRecognizer>(model_file.string(), char_dict_path.string(), option);
            }),
                 pybind11::arg("model_file"),
                 pybind11::arg("char_dict_path") = "",
                 pybind11::arg("option") = pybind11::none())
            .def("predict",
                 [](ocr::FormulaRecognizer& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::string latex;
                     if (!self.predict(ImageData(mat), &latex)) {
                         throw std::runtime_error(
                             "Failed to run FormulaRecognizer.predict.");
                     }
                     return latex;
                 }, pybind11::arg("image"))
            .def("is_initialized", &ocr::FormulaRecognizer::is_initialized);

        // DocToMarkdown: orchestrates layout -> (ocr/table/formula) -> Markdown
        pybind11::class_<ocr::DocToMarkdown>(m, "DocToMarkdown")
            .def(pybind11::init<>())
            .def("set_layout",
                 [](ocr::DocToMarkdown& self, ocr::StructureV2Layout& layout) {
                     self.set_layout(&layout);
                 }, pybind11::arg("layout"), pybind11::keep_alive<1, 2>())
            .def("set_table",
                 [](ocr::DocToMarkdown& self, ocr::PPStructureV2Table& table) {
                     self.set_table(&table);
                 }, pybind11::arg("table"), pybind11::keep_alive<1, 2>())
            .def("set_formula",
                 [](ocr::DocToMarkdown& self, ocr::FormulaRecognizer& formula) {
                     self.set_formula(&formula);
                 }, pybind11::arg("formula"), pybind11::keep_alive<1, 2>())
            .def("set_ocr",
                 [](ocr::DocToMarkdown& self, ocr::PaddleOCR& ocr) {
                     self.set_ocr(&ocr);
                 }, pybind11::arg("ocr"), pybind11::keep_alive<1, 2>())
            .def("predict",
                 [](ocr::DocToMarkdown& self, pybind11::array& image) {
                     const auto mat = pyarray_to_cv_mat(image);
                     std::string markdown;
                     if (!self.predict(ImageData(mat), &markdown)) {
                         throw std::runtime_error(
                             "Failed to run DocToMarkdown.predict.");
                     }
                     return markdown;
                 }, pybind11::arg("image"))
            .def("ready", &ocr::DocToMarkdown::ready);
    }
} // namespace modeldeploy::vision
