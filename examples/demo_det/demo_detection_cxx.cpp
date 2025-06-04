//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(1);
    modeldeploy::vision::detection::UltralyticsDet yolov8("../../test_data/test_models/yolo11n.onnx", option);
    auto img = cv::imread("../../test_data/test_images/test_person.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    // yolov8.get_preprocessor().set_size({1440, 1440});
    int warm_up_count = 0;
    for (int i = 0; i < warm_up_count; ++i) {
        yolov8.predict(img, &result);
    }

    int loop_count = 1;
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < loop_count; ++i) {
        yolov8.predict(img, &result);
    }
    auto end_time = std::chrono::steady_clock::now();
    std::cout << "infer time: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() / loop_count
        << " ms" << std::endl;
    // result.display();
    const auto vis_image =
        modeldeploy::vision::vis_detection(img, result, 0.3, "../../test_data/test_models/msyh.ttc", 12, 0.3, false);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
