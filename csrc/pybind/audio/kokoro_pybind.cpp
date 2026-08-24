//
// Created by aichao on 2025/6/10.
//

#include "pybind/utils/utils.h"
#include "audio/tts/kokoro.h"

namespace modeldeploy::audio {
    void bind_kokoro(pybind11::module& m) {
        pybind11::class_<tts::Kokoro, BaseModel>(m, "Kokoro")
            .def(pybind11::init([](const std::filesystem::path& model_file_path,
                                   const std::filesystem::path& token_path_str,
                                   const std::vector<std::string>& lexicons,
                                   const std::filesystem::path& voices_bin,
                                   const std::filesystem::path& jieba_dir,
                                   const std::filesystem::path& text_normalization_dir,
                                   const RuntimeOption& option) {
                     return std::make_unique<tts::Kokoro>(
                         model_file_path.string(), token_path_str.string(), lexicons,
                         voices_bin.string(), jieba_dir.string(),
                         text_normalization_dir.string(), option);
                 }),
                 pybind11::arg("model_file_path"),
                 pybind11::arg("token_path_str"),
                 pybind11::arg("lexicons"),
                 pybind11::arg("voices_bin"),
                 pybind11::arg("jieba_dir"),
                 pybind11::arg("text_normalization_dir"),
                 pybind11::arg("option"))
            .def("predict",
                 [](tts::Kokoro& self, const std::string& text, const std::string& voice, const float speed) {
                     std::vector<float> out_audio;
                     self.predict(text, voice, speed, &out_audio);
                     return out_audio;
                 }, pybind11::arg("text"), pybind11::arg("voice"), pybind11::arg("speed"))
            .def_property("sample_rate", &tts::Kokoro::get_sample_rate, &tts::Kokoro::set_sample_rate);
    }
} // modeldeploy::audio
