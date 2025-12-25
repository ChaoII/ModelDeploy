//
// Created by aichao on 2025/7/30.
//

#include <filesystem>
#include <thread>
#include "csrc/vision.h"
#include <algorithm>
#include "vision/common/display/display.h"

namespace fs = std::filesystem;

void predict(modeldeploy::vision::detection::UltralyticsDet* model, const int thread_id,
             const std::vector<std::string>& images) {
    thread_local int image_index = 0;
    for (auto const& image_file : images) {
        image_index++;
        auto im = modeldeploy::ImageData::imread(image_file);
        std::vector<modeldeploy::vision::DetectionResult> res;
        if (!model->predict(im, &res)) {
            std::cerr << "Failed to predict." << std::endl;
            return;
        }
        // print res
        std::cout << "Thread Id: " << thread_id << " | image index: " << image_index << " | obj count: " << res.size() <<
            std::endl;
    }
}


void get_image_list(std::vector<std::vector<std::string>>* image_list, const std::string& image_dir,
                    const int thread_num) {
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
    image_list->resize(thread_num);
    const size_t num = count / thread_num;
    for (int i = 0; i < thread_num; ++i) {
        const size_t start = i * num;
        const size_t end = (i == thread_num - 1) ? count : (i + 1) * num;
        (*image_list)[i] = std::vector<std::string>(images.begin() + start, images.begin() + end);
    }
}


int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(10);
    option.use_ort_backend();
    option.use_gpu(0);
    option.enable_fp16 = true;
    option.enable_trt = true;
    option.ort_option.trt_engine_cache_path = "./trt_engine";
    const modeldeploy::vision::detection::UltralyticsDet model(
        "../../test_data/test_models/yolo11n.onnx", option);
    if (!model.is_initialized()) {
        std::cerr << "Failed to initialize model." << std::endl;
        return -1;
    }
    const std::string image_file_path = "F:/ultralytics_workspace/dataset/D000007/split/images/train";
    constexpr int thread_num = 16;
    std::vector<decltype(model.clone())> models;
    models.reserve(thread_num);
    for (int i = 0; i < thread_num; ++i) {
        models.emplace_back(std::move(model.clone()));
    }
    std::vector<std::vector<std::string>> image_list(thread_num);
    get_image_list(&image_list, image_file_path, thread_num);
    const auto start_time = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(thread_num);

    for (int i = 0; i < thread_num; ++i) {
        std::cout << "Thread Id: " << i << " | image count: " << image_list[i].size() << std::endl;
        threads.emplace_back(predict, models[i].get(), i, image_list[i]);
    }

    for (int i = 0; i < thread_num; ++i) {
        threads[i].join();
    }
    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
        << "ms" << std::endl;
}
