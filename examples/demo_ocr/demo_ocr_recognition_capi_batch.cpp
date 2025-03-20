//
// Created by aichao on 2025/2/26.
//

#include <iostream>
#include <fstream>
#ifdef WIN32
#include <windows.h>
#endif
#include <vector>
#include "capi/utils/md_utils_capi.h"
#include "capi/utils/md_image_capi.h"
#include "capi/vision/ocr/ocr_recognition_capi.h"
#include "capi/vision/ocr/ocr_capi.h"

void points_to_polygon(const std::vector<MDPoint>& points, MDPolygon* polygon) {
    polygon->size = static_cast<int>(points.size());
    polygon->data = static_cast<MDPoint*>(malloc(sizeof(MDPoint) * polygon->size));
    for (int i = 0; i < polygon->size; i++) {
        polygon->data[i] = points[i];
    }
}

int main(int argc, char** argv) {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    MDStatusCode ret;
    //简单百宝箱
    MDModel model;
    if ((ret = md_create_ocr_recognition_model(&model,
                                               "../../test_data/test_models/ocr/repsvtr_mobile/rec_infer.onnx",
                                               "../../test_data/key.txt")) != 0) {
        md_free_ocr_recognition_model(&model);
        std::cout << ret << std::endl;
        return ret;
    }
    const std::vector<std::vector<MDPoint>> points = {
        {{1107, 2237}, {1333, 2237}, {1333, 2296}, {1107, 2296}},
        {{1069, 2548}, {1420, 2548}, {1420, 2611}, {1069, 2611}}
    };

    auto polygons = std::vector<MDPolygon>(points.size());
    for (int i = 0; i < points.size(); i++) {
        // 必须获取它的引用
        auto& polygon = polygons.at(i);
        // 此处给polygon开辟了内存，后面用完后一定要记得释放
        points_to_polygon(points[i], &polygon);
    }

    MDImage image = md_read_image("../../test_data/test_images/ocr_check_report1.png");
    MDOCRResults results;
    if ((ret = md_ocr_recognition_model_predict_batch(&model, &image, 8,
                                                      polygons.data(),
                                                      static_cast<int>(polygons.size()),
                                                      &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }

    for (int i = 0; i < results.size; i++) {
        std::cout << "text: " << results.data[i].text << std::endl;
        std::cout << "score: " << results.data[i].score << std::endl;
    }

    // 释放内存
    for (auto& polygon : polygons) {
        md_free_polygon(&polygon);
    }
    md_free_image(&image);
    md_free_ocr_result(&results);
    md_free_ocr_recognition_model(&model);
}
