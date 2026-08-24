#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/utils/utils.h"
#include "nlp/solutions/text_classifier.h"
#include "runtime/runtime_option.h"

namespace modeldeploy {
namespace nlp {

void bind_solutions(pybind11::module& m) {
    pybind11::class_<solution::TextClassifier>(m, "TextClassifier")
        .def(pybind11::init([](const std::filesystem::path& model_file,
                               pybind11::object option_obj) {
                 RuntimeOption option = pybind11::none().equal(option_obj) ? RuntimeOption() : option_obj.cast<RuntimeOption>();
                 return std::make_unique<solution::TextClassifier>(model_file.string(), option);
             }),
             pybind11::arg("model_file"), pybind11::arg("option") = pybind11::none())
        .def("predict", [](solution::TextClassifier& s, const std::string& text) {
            int label; float score;
            if (!s.predict(text, &label, &score)) throw std::runtime_error("TextClassifier predict failed");
            return std::make_pair(label, score);
        })
        .def("is_initialized", &solution::TextClassifier::is_initialized);
}

} // namespace nlp
} // namespace modeldeploy
