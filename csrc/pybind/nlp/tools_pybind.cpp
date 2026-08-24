#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/utils/utils.h"
#include "nlp/tools/tokenizer.h"
#include "nlp/tools/splitter.h"
#include "nlp/tools/keywords.h"
#include "nlp/tools/stats.h"

namespace modeldeploy {
namespace nlp {

void bind_tools(pybind11::module& m) {
    pybind11::class_<tool::Tokenizer>(m, "Tokenizer")
        .def(pybind11::init([](const std::filesystem::path& dict_dir) {
                 return std::make_unique<tool::Tokenizer>(dict_dir.string());
             }),
             pybind11::arg("dict_dir"))
        .def("tokenize", &tool::Tokenizer::tokenize, pybind11::arg("text"), pybind11::arg("mode") = "mix")
        .def("is_loaded", &tool::Tokenizer::is_loaded);
    pybind11::class_<tool::Splitter>(m, "Splitter")
        .def_static("split_sentences", &tool::Splitter::split_sentences);
    pybind11::class_<tool::Keywords>(m, "Keywords")
        .def_static("top", &tool::Keywords::top, pybind11::arg("text"), pybind11::arg("k") = 5);
    pybind11::class_<tool::Stats>(m, "Stats")
        .def_static("char_count", &tool::Stats::char_count)
        .def_static("word_count", &tool::Stats::word_count)
        .def_static("sentence_count", &tool::Stats::sentence_count);
}

} // namespace nlp
} // namespace modeldeploy
