#include <algorithm>

#include "pybind/utils/utils.h"
#include "audio/tts/audio8/audio8.h"

namespace modeldeploy::audio {
    void bind_audio8(pybind11::module& m) {
        pybind11::class_<tts::Audio8, BaseModel>(m, "Audio8")
            .def(pybind11::init([](const std::filesystem::path& model_dir,
                                   const RuntimeOption& option) {
                       auto audio8 = std::make_unique<tts::Audio8>();
                       if (!audio8->Load(model_dir.string(), option)) {
                           throw std::runtime_error("Audio8: failed to load model from " +
                                                    model_dir.string());
                       }
                       return audio8;
                   }),
                 pybind11::arg("model_dir"), pybind11::arg("option"))
            .def("predict", [](tts::Audio8& self, const std::string& text, const std::string& voice,
                               float speed) {
                       std::vector<float> out;
                       self.predict(text, voice, speed, &out);
                       return out;
                   },
                 pybind11::arg("text"), pybind11::arg("voice"), pybind11::arg("speed") = 1.0f)
            .def("predict_stream", [](tts::Audio8& self, const std::string& text,
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
            .def_property_readonly("sample_rate", &tts::Audio8::get_sample_rate);
    }
}  // namespace modeldeploy::audio
