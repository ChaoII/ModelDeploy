#include "runtime_factory.hpp"

#include <filesystem>
#include <string>

using namespace modeldeploy;

RuntimeOption build_runtime_option(const ModelConfig& cfg) {
    RuntimeOption opt;
    if (cfg.device == "gpu") opt.set_device(Device::GPU, 0);
    opt.set_cpu_thread_num(1);

    // 自动识别 .engine 文件 → 走纯 TRT 后端
    const bool is_engine_file = (cfg.path.size() > 7 &&
        (cfg.path.substr(cfg.path.size() - 7) == ".engine" ||
         cfg.path.substr(cfg.path.size() - 7) == ".Engine"));

    if (is_engine_file || cfg.backend == "trt" || cfg.backend == "tensorrt") {
        opt.use_trt_backend();
        opt.enable_fp16 = true;
        std::string cache_dir = "data/trt_cache";
        try { std::filesystem::create_directories(cache_dir); } catch (...) {}
        std::string model_name = cfg.path.substr(cfg.path.find_last_of("/\\") + 1);
        opt.trt_option.cache_file_path = cache_dir + "/" + model_name + ".engine";
        opt.trt_option.enable_fp16 = true;
        opt.trt_option.max_workspace_size = 1ULL << 30;  // 1GB
        if (cfg.input_size.size() == 2) {
            std::string s = "1x3x" + std::to_string(cfg.input_size[0]) + "x" + std::to_string(cfg.input_size[1]);
            opt.set_trt_min_shape(s); opt.set_trt_opt_shape(s); opt.set_trt_max_shape(s);
        }
    } else if (cfg.backend == "mnn") {
        opt.use_mnn_backend();
    } else if (cfg.backend == "sophgo" || cfg.device == "tpu") {
        opt.use_sophgo_backend();
        opt.device_id = 0;  // use_sophgo_backend() 默认 device_id=-1，须显式 0
    } else {
        opt.use_ort_backend();
        if (cfg.device == "gpu") {
            opt.enable_fp16 = true;
            opt.ort_option.enable_fp16 = true;
            // 可选：启用 ORT TensorRT EP（首次会构建 TRT engine，较慢；需本机 TensorRT）
            if (cfg.use_trt_ep) opt.enable_trt = true;
            std::string cache_dir = "data/ort_trt_cache";
            try { std::filesystem::create_directories(cache_dir); } catch (...) {}
            opt.ort_option.trt_engine_cache_path = cache_dir;
        }
    }
    return opt;
}
