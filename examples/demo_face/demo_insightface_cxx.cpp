//
// insightface buffalo_l 全流程 demo（det + 2d106 + 3d68 + recognition）。
// 用法: demo_insightface_cxx <model_dir> <image_path>
//
#include <iostream>
#include <string>
#include "csrc/vision/face/insightface/face_analysis.h"

using namespace modeldeploy::vision;

int main(int argc, char** argv) {
    const std::string model_dir = argc > 1 ? argv[1]
        : "../../test_data/test_models/onnx/insightface/buffalo_l";
    const std::string image_path = argc > 2 ? argv[2]
        : "../../test_data/test_images/test_person.jpg";

    auto analysis = face::InsightFaceAnalysis::create_from_dir(model_dir);
    if (!analysis || !analysis->is_initialized()) {
        std::cerr << "Failed to init insightface models from " << model_dir << std::endl;
        return -1;
    }

    auto img = ImageData::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    std::vector<face::InsightFaceResult> results;
    if (!analysis->analyze(img, &results)) {
        std::cerr << "Analyze failed" << std::endl;
        return -1;
    }

    std::cout << "detected faces: " << results.size() << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "face" << i << ":" << std::endl;
        std::cout << "  bbox=[" << r.bbox[0] << ", " << r.bbox[1] << ", "
                  << r.bbox[2] << ", " << r.bbox[3] << "] score=" << r.det_score << std::endl;
        if (!r.kps.empty()) {
            std::cout << "  kps[0]=" << r.kps[0][0] << ", " << r.kps[0][1] << std::endl;
        }
        if (!r.landmark_2d_106.empty()) {
            std::cout << "  2d106[0]=" << r.landmark_2d_106[0][0] << ", " << r.landmark_2d_106[0][1] << std::endl;
        }
        if (!r.landmark_3d_68.empty()) {
            std::cout << "  3d68[0]=" << r.landmark_3d_68[0][0] << ", " << r.landmark_3d_68[0][1]
                      << ", " << r.landmark_3d_68[0][2] << std::endl;
        }
        std::cout << "  pose=" << r.pose[0] << ", " << r.pose[1] << ", " << r.pose[2] << std::endl;
        if (!r.embedding.empty()) {
            std::cout << "  emb[:3]=" << r.embedding[0] << ", " << r.embedding[1]
                      << ", " << r.embedding[2] << " dim=" << r.embedding.size() << std::endl;
        }
    }
    return 0;
}
