#include "capability.hpp"
#include <fstream>

nlohmann::json detect_capabilities(int max_channels) {
    using nlohmann::json;
    json backends = json::array();
#ifdef ENABLE_ORT
    backends.push_back("ort");
#endif
#ifdef ENABLE_MNN
    backends.push_back("mnn");
#endif
#ifdef ENABLE_TRT
    backends.push_back("trt");
#endif
#ifdef ENABLE_NCNN
    backends.push_back("ncnn");
#endif
#ifdef ENABLE_SOPHGO
    backends.push_back("sophgo");
#endif

    std::string platform = "cpu";
#if defined(ENABLE_SOPHGO)
    platform = "sophgo";
#elif defined(WITH_GPU)
#if defined(__linux__)
    if (std::ifstream("/etc/nv_tegra_release").good()) platform = "jetson";
    else
#endif
    platform = "nvidia";
#endif

    json families = json::array({"det", "cls", "face"});

    json decode = json::array({"h264", "h265"});
    json encode = json::array({"h264"});
    json hw = json::array();
#if defined(WITH_GPU)
    hw.push_back("nvenc");
    hw.push_back("nvdec");
#endif
#if defined(ENABLE_VAAPI)
    hw.push_back("vaapi");
#endif

    json j;
    j["hardware"] = {{"platform", platform}, {"gpu_model", platform == "nvidia" ? "nvidia" : ""}, {"vram_mb", 0}};
    j["backends"] = backends;
    j["model_families"] = families;
    j["max_channels"] = max_channels;
    j["codecs"] = {{"decode", decode}, {"encode", encode}, {"hw", hw}};
    return j;
}
