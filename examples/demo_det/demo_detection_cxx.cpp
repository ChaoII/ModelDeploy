//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"
#include <MNN/expr/Executor.hpp>
#include <MNN/Interpreter.hpp>

void test_camera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Cannot open camera" << std::endl;
        return;
    }

    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_trt = true;
    modeldeploy::vision::detection::UltralyticsDet yolo11_det("../../test_data/test_models/yolo11n.onnx", option);
    std::vector<modeldeploy::vision::DetectionResult> result;
    // yolov8.get_preprocessor().set_size({1440, 1440});
    yolo11_det.get_preprocessor().set_mini_pad(true);

    cv::Mat frame;
    while (true) {
        if (cap.read(frame)) {
            std::cout << "width:" << frame.cols << " height:" << frame.rows << std::endl;
            yolo11_det.predict(frame, &result);
            auto r = modeldeploy::vision::vis_det(frame, result);
            cv::imshow("frame", r);
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    }
}


int main() {

    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(6);
    option.use_mnn_backend();
    option.use_gpu(0);
    // option.enable_fp16 = true;
    // option.enable_trt = true;
    modeldeploy::vision::detection::UltralyticsDet yolo11_det("../../test_data/test_models/yolo11n.mnn", option);
    auto img = cv::imread("../../test_data/test_images/test_person.jpg");
    std::vector<modeldeploy::vision::DetectionResult> result;
    // yolo11_det.get_preprocessor().set_size({640, 640});
    int warming_up_count = 10;
    for (int i = 0; i < warming_up_count; ++i) {
        yolo11_det.predict(img, &result);
    }
    TimerArray timers;
    int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        yolo11_det.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    const auto vis_image =
        modeldeploy::vision::vis_det(img, result, 0.3, "../../test_data/msyh.ttc", 12, 0.3,
                                     true);
    cv::imshow("test", vis_image);
    cv::waitKey(0);
    // test_camera();
}
