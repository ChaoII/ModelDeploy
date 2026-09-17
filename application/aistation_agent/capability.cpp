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

    // 模型族上报须与实际接线一致：obb/iseg/pose/face_attr/face_as/depth/sem/face_landmark/
    // doc/barcode/sam/face_rec 已接入推理与事件（见 pipeline/inference_engine），名称与云端
    // 场景目录 catalog.py 的声明对齐：type="obb"/"iseg"/"pose"/"face_attr"/"face_as"/"depth"/
    // "sem"/"face_landmark"/"doc"/"barcode"/"sam"/"face_rec"。
    // face_attr → 人脸框 attributes{gender_male, age_young}；face_as → attributes{liveness}；
    // depth → objects[].depth（米）；sem → 整帧框 attributes{类别名: 占比}；
    // face_landmark → objects[].keypoints（106 点，归一化）；doc/barcode → objects[].text；
    // sam（SAM_SEG）→ 提示点选段后的单个 segment 框（label="segment"）；
    // face_rec（FACE_REC/STRANGER）→ 人脸框 objects[].embedding（512-d，已 L2 归一化）。
    json families = json::array({"det", "cls", "face", "pedestrian_attribute", "ocr", "lpr",
                                 "tracking", "obb", "iseg", "pose", "face_attr", "face_as",
                                 "depth", "sem", "face_landmark", "doc", "barcode",
                                 "sam", "face_rec"});

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
