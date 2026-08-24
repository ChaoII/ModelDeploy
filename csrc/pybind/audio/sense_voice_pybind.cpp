#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind/utils/utils.h"
#include "audio/asr/sense_voice.h"

namespace modeldeploy {
namespace audio {

void bind_sense_voice(pybind11::module& m) {
    pybind11::class_<asr::SenseVoiceResult>(m, "SenseVoiceResult")
        .def_readwrite("text", &asr::SenseVoiceResult::text)
        .def_readwrite("language", &asr::SenseVoiceResult::language)
        .def_readwrite("emotion", &asr::SenseVoiceResult::emotion)
        .def_readwrite("event", &asr::SenseVoiceResult::event)
        .def_readwrite("task", &asr::SenseVoiceResult::task)
        .def_readwrite("itn", &asr::SenseVoiceResult::itn)
        .def_readwrite("nospeech", &asr::SenseVoiceResult::nospeech);

    pybind11::class_<asr::SenseVoice>(m, "SenseVoice")
        .def(pybind11::init([](const std::filesystem::path& model_file,
                               const std::filesystem::path& token_path_str,
                               pybind11::object custom_option_obj) {
                 RuntimeOption custom_option = pybind11::none().equal(custom_option_obj) ? RuntimeOption() : custom_option_obj.cast<RuntimeOption>();
                 return std::make_unique<asr::SenseVoice>(
                     model_file.string(), token_path_str.string(), custom_option);
             }),
             pybind11::arg("model_file"), pybind11::arg("token_path_str"),
             pybind11::arg("custom_option") = pybind11::none())
        .def("name", &asr::SenseVoice::name)
        // 文本结果
        .def("predict", [](asr::SenseVoice& sv, const std::vector<float>& data) {
                 std::string text;
                 if (!sv.predict(data, &text)) throw std::runtime_error("SenseVoice predict failed");
                 return text;
             }, pybind11::arg("data"))
        // 结构化结果
        .def("recognize", [](asr::SenseVoice& sv, const std::vector<float>& data) {
                 asr::SenseVoiceResult r;
                 if (!sv.predict(data, &r)) throw std::runtime_error("SenseVoice predict failed");
                 return r;
             }, pybind11::arg("data"));
}

} // namespace audio
} // namespace modeldeploy
