//
// Created by aichao on 2025/6/10.
//

#include "csrc/pybind/utils/utils.h"
#include "csrc/audio/tts/kokoro.h"

namespace modeldeploy::audio {
    void bind_kokoro(pybind11::module& m) {
        pybind11::class_<tts::Kokoro, BaseModel>(m, "Kokoro")
            .def(pybind11::init<const std::string&, const std::string&,
                                const std::vector<std::string>&, const std::string&,
                                const std::string&, const std::string&, const RuntimeOption&>())
            .def("predict",
                 [](tts::Kokoro& self, const std::string& text, const std::string& voice, const float speed) {
                     std::vector<float> out_audio;
                     self.predict(text, voice, speed, &out_audio);
                     return out_audio;
                 })

            .def_property("sample_rate", &tts::Kokoro::get_sample_rate, &tts::Kokoro::set_sample_rate);
    }
} // modeldeploy::audio
