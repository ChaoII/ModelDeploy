//
// Created by aichao on 2025/2/26.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/pipeline/pedestrian_attribute_capi.h"

int main(int argc, char** argv) {
    MDStatusCode ret;
    //简单百宝箱
    MDModel model;

    MDRuntimeOption option = md_create_default_runtime_option();
    option.device = MD_DEVICE_GPU;
    option.backend = MD_BACKEND_TRT;
    option.enable_trt = 1;
    option.enable_fp16 = 1;
    if ((ret = md_create_attr_model(&model, "../../test_data/test_models/zhgd_det.engine",
                                    "../../test_data/test_models/zhgd_ml.engine", &option)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_set_attr_cls_batch_size(&model, 8);
    md_set_attr_cls_input_size(&model, {192, 256});
    md_set_attr_det_input_size(&model, {1280, 1280});
    md_set_attr_det_threshold(&model, 0.5);
    MDImage image = md_read_image("../../test_data/test_images/test_pedestrian_attribute.jpg");
    MDAttributeResults results;
    if ((ret = md_attr_model_predict(&model, &image, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }

    MDMapData label_map;
    const auto l1 = md_create_key_value_pair(0, "张三");
    const auto l2 = md_create_key_value_pair(1, "李四");
    const auto l3 = md_create_key_value_pair(2, "王五");
    const auto l4 = md_create_key_value_pair(3, "赵六");

    label_map.size = 4;
    label_map.data = static_cast<MDKeyValuePair*>(malloc(sizeof(MDKeyValuePair) * label_map.size));
    label_map.data[0] = l1;
    label_map.data[1] = l2;
    label_map.data[2] = l3;
    label_map.data[3] = l4;

    md_draw_attr_result(&image, &results, 0.5, &label_map, "../../test_data/msyh.ttc", 14, 0.5, 1);
    md_print_attr_result(&results);
    md_free_attr_result(&results);
    md_show_image(&image);
    // 释放内存
    md_free_image(&image);
    md_free_attr_model(&model);
    md_free_md_map(&label_map);
    return ret;
}
