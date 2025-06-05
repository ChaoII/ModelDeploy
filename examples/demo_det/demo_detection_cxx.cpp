//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(1);
    // option.use_gpu();
    modeldeploy::vision::detection::UltralyticsDet yolo11_det("../../test_data/test_models/yolo11n.onnx", option);
    auto img = cv::imread("../../test_data/test_images/test_person.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    // yolov8.get_preprocessor().set_size({1440, 1440});
    yolo11_det.get_preprocessor().set_mini_pad(true);
    int warm_up_count = 1;
    for (int i = 0; i < warm_up_count; ++i) {
        yolo11_det.predict(img, &result);
    }

    TimerArray timers;
    int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        yolo11_det.predict(img, &result, &timers);
    }
    print_benchmark(timers);

    const auto vis_image =
        modeldeploy::vision::vis_detection(img, result, 0.3, "../../test_data/test_models/msyh.ttc", 12, 0.3,
                                           false);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
}
