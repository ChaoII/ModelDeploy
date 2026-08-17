//
// Created by aichao on 2025/2/24.
//

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"


int main(int argc, char** argv) {
    const std::string det_model = argc > 1 ? argv[1]
        : "../../test_data/test_models/onnx/zhgd_det.onnx";
    const std::string ml_model = argc > 2 ? argv[2]
        : "../../test_data/test_models/onnx/zhgd_ml.onnx";
    const std::string image_path = argc > 3 ? argv[3]
        : "../../test_data/test_images/test_pedestrian_attribute_scale.png";
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    modeldeploy::vision::pipeline::PedestrianAttribute pedestrian_attribute(
        det_model, ml_model, option);
    auto img = modeldeploy::vision::ImageData::imread(image_path);
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
        modeldeploy::vision::vis_attr(img, results, 0.5, label_map, "../../test_data/msyh.ttc", 6, 0.15, true, {0, 1});
    (void)vis_image.imwrite("pedestrian_attr_out.jpg");
}
