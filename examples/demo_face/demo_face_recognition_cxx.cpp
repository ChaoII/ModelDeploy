//
// Created by aichao on 2025/3/25.
//

#include "seeta/FaceRecognizer.h"
#include "Struct_cv.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

int main_image() {
    seeta::cv::ImageData image = cv::imread("../../test_data/test_images/test_face_recognition.png");
    seeta::Rect face(97, 54, 127, 170);
    std::vector<SeetaPointF> points = {
        {134.929, 126.889},
        {190.865, 120.054},
        {167.091, 158.991},
        {143.787, 186.269},
        {193.805, 181.186},

    };

    seeta::ModelSetting setting;
    setting.set_id(0);
    setting.append("../../test_data/test_models/seetaface/face_recognizer.csta");

    setting.set_device(SEETA_DEVICE_CPU);
    seeta::FaceRecognizer FR_cpu(setting);

    std::cout << "Got image: [" << image.width << ", " << image.height << ", " << image.channels << "]" << std::endl;

    // auto patch = FR_cpu.CropFaceV2(image, points.data());
    seeta::cv::ImageData patch = cv::imread("vis_result.jpg");

    std::cout << patch.width << std::endl;
    std::cout << patch.height << std::endl;

    cv::imwrite("patch.png", seeta::cv::ImageData(patch).toMat());

    std::shared_ptr<float> features_cpu(new float[FR_cpu.GetExtractFeatureSize()]);

    FR_cpu.ExtractCroppedFace(patch, features_cpu.get());

    auto& FR = FR_cpu;
    auto& features = features_cpu;


    int N = 100;
    std::cout << "Compute " << N << " times. " << std::endl;

    using namespace std::chrono;
    microseconds duration(0);
    for (int i = 0; i < N; ++i) {
        if (i % 10 == 0) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        FR.ExtractCroppedFace(patch, features.get());
        auto end = system_clock::now();
        duration += duration_cast<microseconds>(end - start);
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;

    float min = 999.0f;
    float max = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < FR.GetExtractFeatureSize(); ++i) {
        if (features.get()[i] < min) min = features.get()[i];
        if (features.get()[i] > max) max = features.get()[i];
        sum += features.get()[i];
        std::cout << features.get()[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
    std::cout << "mean: " << sum / 1024 << std::endl;

    cv::waitKey();

    return 0;
}

int main_video() {
    return 0;
}

int main() {
    return main_image();
}
