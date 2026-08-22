#include <opencv2/opencv.hpp>
#include <iostream>
#include "vision/common/image_data.h"
#include "vision/barcode/barcode.h"

// 将 cv::Mat 转 ImageData（ImageData 有 explicit ImageData(const cv::Mat&)）
static modeldeploy::vision::ImageData mat_to_image(const cv::Mat& img) {
    return modeldeploy::vision::ImageData(img);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: demo_barcode <image>\n"; return 1; }
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) { std::cerr << "failed to read " << argv[1] << "\n"; return 1; }
    auto image = mat_to_image(img);
    modeldeploy::vision::barcode::BarcodeDetector det;
    auto res = det.detect(image);
    if (res.empty()) { std::cout << "no barcode found\n"; return 0; }
    for (auto& r : res) {
        std::cout << "[" << r.format << "] " << r.text
                  << " (score=" << r.score << ", is_qr=" << r.is_qr << ")\n";
    }
    return 0;
}
