#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/reid/reid.h"
#include "vision/reid/gallery.h"

// 将 cv::Mat 转 ImageData（ImageData 有 explicit ImageData(const cv::Mat&)）
static modeldeploy::vision::ImageData mat_to_image(const cv::Mat& img) {
    return modeldeploy::vision::ImageData(img);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: demo_reid <osnet_model> <imgA> <imgB>\n";
        std::cerr << "  加载 OSNet 提取两张裁剪行人图的 512-d embedding，入 ReIdGallery 后 match.\n";
        return 1;
    }
    const std::string model_file = argv[1];
    const std::string imgA_file = argv[2];
    const std::string imgB_file = argv[3];

    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    option.set_device(modeldeploy::Device::CPU);

    modeldeploy::vision::reid::ReID reid(model_file, option);
    if (!reid.is_initialized()) {
        std::cerr << "failed to init ReID model: " << model_file << "\n";
        return 1;
    }

    cv::Mat matA = cv::imread(imgA_file);
    cv::Mat matB = cv::imread(imgB_file);
    if (matA.empty() || matB.empty()) {
        std::cerr << "failed to read images: " << imgA_file << " / " << imgB_file << "\n";
        return 1;
    }

    auto embed = [&](const cv::Mat& mat, const char* tag) -> std::vector<float> {
        auto image = mat_to_image(mat);
        std::vector<modeldeploy::vision::ReIdResult> res;
        if (!reid.predict(image, &res) || res.empty() || res[0].embedding.empty()) {
            std::cerr << "prediction failed for " << tag << "\n";
            return {};
        }
        std::cout << tag << " embedding dim=" << res[0].embedding.size() << "\n";
        return res[0].embedding;
    };

    auto eA = embed(matA, "imgA");
    auto eB = embed(matB, "imgB");
    if (eA.empty() || eB.empty()) { return 1; }

    modeldeploy::vision::reid::ReIdGallery gallery;
    gallery.enroll("A", eA);
    gallery.enroll("B", eB);
    std::cout << "gallery size=" << gallery.size() << "\n";

    auto top = gallery.match(eB, 1);
    for (auto& [label, score] : top) {
        std::cout << "match(imgB, 1) -> label=" << label << " score=" << score << "\n";
    }
    return 0;
}
