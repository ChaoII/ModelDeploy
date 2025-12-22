//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


int main() {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.enable_trt = true;
    option.enable_fp16 = true;
    option.use_trt_backend();
    modeldeploy::vision::pipeline::PedestrianAttribute pedestrian_attribute(
        "../../test_data/test_models/zhgd_det_20251219.engine",
        "../../test_data/test_models/zhgd_ml.engine", option);
    auto img = modeldeploy::ImageData::imread("../../test_data/test_images/test_pedestrian_attribute.jpg");
    pedestrian_attribute.set_cls_batch_size(8);
    pedestrian_attribute.set_det_input_size({1280, 1280});
    pedestrian_attribute.set_det_threshold(0.5);
    pedestrian_attribute.set_cls_input_size({192, 256});
    std::vector<modeldeploy::vision::AttributeResult> results;
    TimerArray timers;
    int loop_count = 100;
    for (int i = 0; i < loop_count; i++) {
        pedestrian_attribute.predict(img, &results, &timers);
        // std::cout << i << "th loop" << std::endl;
    }
    timers.print_benchmark();
    modeldeploy::vision::dis_attr(results);
    std::unordered_map<int, std::string> label_map;
    label_map.insert({0, "safety_helmet"});
    label_map.insert({1, "reflective_vest"});
    label_map.insert({2, "safety_rope"});
    label_map.insert({3, "work_uniform"});
    const auto vis_image =
        modeldeploy::vision::vis_attr(img, results, 0.5, label_map, "../../test_data/msyh.ttc", 6, 0.15, true,{0,1});
    vis_image.imshow("pedestrian");
}