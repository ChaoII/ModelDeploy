//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"


void test_camera() {
    // cv::VideoCapture cap(0);
    // if (!cap.isOpened()) {
    //     std::cout << "Cannot open camera" << std::endl;
    //     return;
    // }
    //
    // modeldeploy::RuntimeOption option;
    // option.use_gpu();
    // option.enable_trt = true;
    // modeldeploy::vision::detection::UltralyticsDet yolo11_det("../../test_data/test_models/yolo11n.onnx", option);
    // std::vector<modeldeploy::vision::DetectionResult> result;
    // // yolov8.get_preprocessor().set_size({1440, 1440});
    // yolo11_det.get_preprocessor().set_mini_pad(true);
    //
    // cv::Mat frame;
    // while (true) {
    //     if (cap.read(frame)) {
    //         std::cout << "width:" << frame.cols << " height:" << frame.rows << std::endl;
    //         yolo11_det.predict(frame, &result);
    //         auto r = modeldeploy::vision::vis_det(frame, result);
    //         cv::imshow("frame", r);
    //         if (cv::waitKey(1) == 'q') {
    //             break;
    //         }
    //     }
    // }
}


int main() {
    modeldeploy::RuntimeOption option;
    option.set_cpu_thread_num(10);
    option.use_ort_backend();
    option.use_gpu(0);
    option.password = "123456";
    option.enable_fp16 = true;
    option.enable_trt = true;
    option.ort_option.trt_engine_cache_path = "./trt_engine";
    modeldeploy::vision::detection::UltralyticsDet yolo11_det("../../test_data/test_models/best.onnx",
                                                              option);
    const auto label_map = yolo11_det.get_label_map("names");
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/best_0.jpg");
    // auto img1 = img.clone();
    std::vector<modeldeploy::vision::DetectionResult> result;
    // yolo11_det.get_preprocessor().use_cuda_preproc();
    yolo11_det.get_preprocessor().set_size({960, 960});
    constexpr int warming_up_count = 10;
    for (int i = 0; i < warming_up_count; ++i) {
        yolo11_det.predict(img, &result);
    }
    TimerArray timers;
    constexpr int loop_count = 100;
    for (int i = 0; i < loop_count; ++i) {
        yolo11_det.predict(img, &result, &timers);
    }
    timers.print_benchmark();
    const auto vis_image =
        modeldeploy::vision::vis_det(img, result, 0.3, label_map, "../../test_data/msyh.ttc", 12, 0.3,
                                     true);
    img.imshow("result");
    // test_camera();
}
