#include <algorithm>

#include "pybind/utils/utils.h"
#include "audio/tts/qwen3/qwen3_tts.h"

namespace modeldeploy::audio {
    void bind_qwen3_tts(pybind11::module& m) {
        pybind11::class_<tts::Qwen3Tts, BaseModel>(m, "Qwen3Tts")
            .def(pybind11::init([](const std::filesystem::path& model_dir,
                                   const RuntimeOption& option) {
                       return new tts::Qwen3Tts(model_dir.string(), option);
                   }),
                 pybind11::arg("model_dir"), pybind11::arg("option"))
            .def("predict", [](tts::Qwen3Tts& self, const std::string& text,
                               const std::string& voice, float speed) {
                       std::vector<float> out;
                       self.predict(text, voice, speed, &out);
                       return out;
                   },
                 pybind11::arg("text"), pybind11::arg("voice"), pybind11::arg("speed") = 1.0f)
            .def("predict_stream", [](tts::Qwen3Tts& self, const std::string& text,
                                      const std::string& voice, float speed, int chunk_frames) {
                       pybind11::list chunks;
                       self.predict_stream(text, voice, speed, chunk_frames,
                            [&](const float* s, int n, float) {
                                std::vector<float> copy(s, s + n);
                                pybind11::array_t<float> arr(copy.size());
                                std::copy(copy.begin(), copy.end(), arr.mutable_data());
                                chunks.append(arr);
                                return true;
                            });
                       return chunks;
                   },
                 pybind11::arg("text"), pybind11::arg("voice"), pybind11::arg("speed") = 1.0f,
                 pybind11::arg("chunk_frames") = 24)
            .def("clone", [](tts::Qwen3Tts& self, const std::string& text,
                             const std::string& ref_audio, const std::string& ref_text,
                             const std::string& lang) {
                       std::vector<float> out;
                       self.clone(text, ref_audio, ref_text, lang, &out);
                       return out;
                   },
                 pybind11::arg("text"), pybind11::arg("ref_audio"), pybind11::arg("ref_text"),
                 pybind11::arg("lang"))
            .def_property_readonly("sample_rate", &tts::Qwen3Tts::get_sample_rate);
    }
}  // namespace modeldeploy::audio
