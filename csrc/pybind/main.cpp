//
// Created by aichao on 2025/6/9.
//

#include <pybind11/pybind11.h>


namespace modeldeploy::vision {
    void bind_vision(pybind11::module&);
}

namespace modeldeploy::audio {
    void bind_kokoro(pybind11::module&);
    void bind_speaker_verify(pybind11::module&);
    void bind_sense_voice(pybind11::module&);
    void bind_solutions(pybind11::module&);
    void bind_tools(pybind11::module&);
}

namespace modeldeploy::nlp {
    void bind_tools(pybind11::module&);
    void bind_solutions(pybind11::module&);
}


namespace modeldeploy::pipeline {
    void bind_pipeline(pybind11::module&);
}

#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
namespace modeldeploy::video {
    void bind_video(pybind11::module&);
}
#endif

namespace modeldeploy {
    void bind_tensor(pybind11::module&);
    void bind_runtime(pybind11::module&);
    void bind_base_model(pybind11::module&);

    PYBIND11_MODULE(modeldeploy, m) {
        m.doc() =
            "Make programmer easier to deploy deeplearning model, save time to save "
            "the world!";
        m.def("get_version", []() { return std::string(MD_VERSION); }, "Get version of modeldeploy.");
        bind_tensor(m);
        bind_runtime(m);
        bind_base_model(m);

        auto pipeline_module =
            m.def_submodule("pipeline", "Pipeline DAG module of Modeldeploy.");
        pipeline::bind_pipeline(pipeline_module);

#if defined(BUILD_VIDEO) && defined(BUILD_VISION)
        auto video_module =
            m.def_submodule("video", "Video decode module of Modeldeploy.");
        video::bind_video(video_module);
#endif

#ifdef BUILD_VISION
        auto vision_module =
            m.def_submodule("vision", "Vision module of Modeldeploy.");
        vision::bind_vision(vision_module);
#endif

#ifdef BUILD_AUDIO
        auto audio_module =
            m.def_submodule("audio", "Audio module of Modeldeploy.");
        audio::bind_kokoro(audio_module);
        audio::bind_speaker_verify(audio_module);
        audio::bind_sense_voice(audio_module);
        auto audio_solutions_m = audio_module.def_submodule("solutions", "Audio solutions");
        auto audio_tools_m = audio_module.def_submodule("tools", "Audio tools");
        audio::bind_solutions(audio_solutions_m);
        audio::bind_tools(audio_tools_m);
#endif

#ifdef BUILD_NLP
        auto nlp_module = m.def_submodule("nlp", "NLP module of Modeldeploy.");
        auto nlp_solutions_m = nlp_module.def_submodule("solutions", "NLP solutions");
        auto nlp_tools_m = nlp_module.def_submodule("tools", "NLP tools");
        nlp::bind_tools(nlp_tools_m);
        nlp::bind_solutions(nlp_solutions_m);
#endif
    }
}
