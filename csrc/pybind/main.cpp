//
// Created by aichao on 2025/6/9.
//

#include <pybind11/pybind11.h>


namespace modeldeploy::vision {
    void bind_vision(pybind11::module&);
}

namespace modeldeploy::audio {
    void bind_kokoro(pybind11::module&);
}


namespace modeldeploy {
    void bind_tensor(pybind11::module&);
    void bind_runtime(pybind11::module&);
    void bind_base_model(pybind11::module&);

    PYBIND11_MODULE(modeldeploy, m) {
        m.doc() =
            "Make programmer easier to deploy deeplearning model, save time to save "
            "the world!";
        bind_tensor(m);
        bind_runtime(m);
        bind_base_model(m);

#ifdef BUILD_VISION
        auto vision_module =
            m.def_submodule("vision", "Vision module of Modeldeploy.");
        vision::bind_vision(vision_module);
#endif

#ifdef BUILD_AUDIO
        auto audio_module =
            m.def_submodule("audio", "Audio module of Modeldeploy.");
        audio::bind_kokoro(audio_module);
#endif
    }
}
