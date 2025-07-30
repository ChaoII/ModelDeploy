//
// Created by aichao on 2025/7/30.
//

#include <filesystem>
#include <thread>
#include "csrc/vision.h"
#include "vision/common/display/display.h"

namespace fs = std::filesystem;

void predict(modeldeploy::vision::detection::UltralyticsDet* model, const int thread_id,
             const std::vector<std::string>& images) {
    for (auto const& image_file : images) {
        auto im = modeldeploy::ImageData::imread(image_file);
        std::vector<modeldeploy::vision::DetectionResult> res;
        if (!model->predict(im, &res)) {
            std::cerr << "Failed to predict." << std::endl;
            return;
        }
        // print res
        std::cout << "Thread Id: " << thread_id << " | obj count: " << res.size() << std::endl;
    }
}


std::vector<std::vector<std::string>> get_image_list(const std::string& image_dir, const int batch_size) {
    std::vector<std::string> images;
    // 遍历目录，收集图片路径（可根据需要添加扩展名筛选）
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            // 可选：只保留图片格式
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                images.push_back(path);
            }
        }
    }
    std::cout << "Total images: " << images.size() << std::endl;
    std::sort(images.begin(), images.end()); // 可选：保持顺序一致
    const size_t count = images.size();
    const size_t num = count / batch_size;
    std::cout << "Total batches: " << num << std::endl;
    std::vector<std::vector<std::string>> image_list;
    for (int i = 0; i < num; ++i) {
        const int start = i * batch_size;
        const int end = start + batch_size;
        image_list.emplace_back(images.begin() + start, images.begin() + end);
    }
    return image_list;
}

std::vector<modeldeploy::ImageData> get_images(const std::vector<std::string>& images_strs) {
    std::vector<modeldeploy::ImageData> images;
    images.reserve(images_strs.size());
    for (auto& images_str : images_strs) {
        images.push_back(modeldeploy::ImageData::imread(images_str));
    }
    return images;
}


int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(10);
    option.use_ort_backend();
    option.use_gpu(0);
    option.password = "123456";
    option.enable_fp16 = true;
    option.enable_trt = true;
    option.ort_option.trt_min_shape = "images:1x3x224x224";
    option.ort_option.trt_opt_shape = "images:8x3x640x640";
    option.ort_option.trt_max_shape = "images:16x3x1280x1280";
    option.ort_option.trt_engine_cache_path = "./trt_engine";
    modeldeploy::vision::detection::UltralyticsDet model(
        "../../test_data/test_models/helmet.onnx", option);
    const std::string image_file_path = "F:/ultralytics_workspace/dataset/D000007/split/images/train";
    constexpr int batch_size = 1;
    const auto image_lists = get_image_list(image_file_path, batch_size);
    const auto start_time = std::chrono::steady_clock::now();
    for (auto& image_list : image_lists) {
        std::cout << "Image list size: " << image_list.size() << std::endl;
        std::vector<std::vector<modeldeploy::vision::DetectionResult>> ress;
        auto images = get_images(image_list);
        std::cout << "Start predict..." << std::endl;
        model.batch_predict(images, &ress);
    }
    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        << "ms" << std::endl;
}
