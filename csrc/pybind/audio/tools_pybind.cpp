#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "audio/tools/resampler.h"
#include "audio/tools/fbank.h"
#include "audio/tools/waveform.h"
#include "audio/tools/itn.h"
#include "audio/tools/itn_engine.h"
#include "audio/tools/hotword.h"
#include "audio/tools/context_graph.h"

namespace modeldeploy {
namespace audio {

void bind_tools(pybind11::module& m) {
    pybind11::class_<tool::Resampler>(m, "Resampler")
        .def_static("resample", &tool::Resampler::resample);
    pybind11::class_<tool::Fbank>(m, "Fbank")
        .def(pybind11::init<int, int>(), pybind11::arg("sample_rate") = 16000, pybind11::arg("num_bins") = 80)
        .def("compute", &tool::Fbank::compute);
    pybind11::class_<tool::Spectrum>(m, "Spectrum")
        .def(pybind11::init<int>(), pybind11::arg("fft_n") = 1024)
        .def("magnitudes", &tool::Spectrum::magnitudes);

    // ── ITN ──────────────────────────────────────────────
    pybind11::class_<tool::InverseTextNormalizer>(m, "InverseTextNormalizer")
        .def(pybind11::init<>())
        .def("normalize",
             static_cast<std::string (tool::InverseTextNormalizer::*)(const std::string&) const>(
                 &tool::InverseTextNormalizer::normalize),
             pybind11::arg("text"), "口读 -> 书面（轻量实现）");

    pybind11::enum_<tool::ItnBackend>(m, "ItnBackend")
        .value("Lightweight", tool::ItnBackend::Lightweight)
        .value("WeText", tool::ItnBackend::WeText);
    pybind11::class_<tool::ItnEngine>(m, "ItnEngine")
        // 枚举默认值无法被 stubgen 解析，改用 none 兜底（与 RuntimeOption 一致）
        .def(pybind11::init([](pybind11::object backend_obj) {
                 tool::ItnBackend backend = pybind11::none().equal(backend_obj)
                                                ? tool::ItnBackend::Lightweight
                                                : backend_obj.cast<tool::ItnBackend>();
                 return std::make_unique<tool::ItnEngine>(backend);
             }),
             pybind11::arg("backend") = pybind11::none())
        .def("normalize", &tool::ItnEngine::normalize, pybind11::arg("text"))
        .def_property_readonly("backend", &tool::ItnEngine::backend);

    // ── 热词 boosting ────────────────────────────────────
    pybind11::class_<tool::FoundHotword>(m, "FoundHotword")
        .def_readonly("word", &tool::FoundHotword::word)
        .def_readonly("weight", &tool::FoundHotword::weight)
        .def_readonly("count", &tool::FoundHotword::count);
    pybind11::class_<tool::Hypothesis>(m, "Hypothesis")
        .def(pybind11::init<>())
        .def_readwrite("text", &tool::Hypothesis::text)
        .def_readwrite("score", &tool::Hypothesis::score);
    pybind11::class_<tool::HotwordContext>(m, "HotwordContext")
        .def(pybind11::init<>())
        .def("add", &tool::HotwordContext::add, pybind11::arg("word"), pybind11::arg("weight") = 1.0f)
        .def("clear", &tool::HotwordContext::clear)
        .def("words", &tool::HotwordContext::words)
        .def("scan", &tool::HotwordContext::scan, pybind11::arg("text"))
        .def("highlight", &tool::HotwordContext::highlight,
             pybind11::arg("text"), pybind11::arg("left") = "[", pybind11::arg("right") = "]");
    m.def("tokenize_chars", &tool::tokenize_chars, pybind11::arg("text"),
          "把文本按 Unicode 码点拆成字符级 token id");
    // 不透明类型：仅作为 build_context_graph 的返回持有者
    pybind11::class_<tool::ContextGraph, std::shared_ptr<tool::ContextGraph>>(m, "ContextGraph");
    m.def("build_context_graph", &tool::build_context_graph,
          pybind11::arg("ctx"), pybind11::arg("tokenize"), pybind11::arg("context_score") = 1.0f,
          "由热词列表构建 ContextGraph（Kaldi/sherpa-onnx 移植）");
    m.def("hotword_score", &tool::score,
          pybind11::arg("graph"), pybind11::arg("tokenize"), pybind11::arg("text"),
          "用 ContextGraph 对文本打分（偏置后得分）");
    m.def("rescore", &tool::rescore,
          pybind11::arg("hyps"), pybind11::arg("ctx"), pybind11::arg("tokenize"),
          pybind11::arg("lambda") = 1.0f,
          "对 n-best 候选按 声学得分 + λ*热词偏置 重排序");
}

} // namespace audio
} // namespace modeldeploy
