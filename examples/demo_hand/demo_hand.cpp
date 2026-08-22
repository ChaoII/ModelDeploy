#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/hand/hand.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: demo_hand <model.onnx> <image.jpg>\n");
        return 1;
    }
    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    modeldeploy::vision::hand::HandKeypoint model(argv[1], opt);
    auto im = cv::imread(argv[2]);
    if (im.empty()) {
        printf("cannot read image %s\n", argv[2]);
        return 1;
    }
    modeldeploy::vision::ImageData img(im);
    std::vector<modeldeploy::vision::KeyPointsResult> res;
    if (!model.predict(img, &res)) {
        printf("predict failed\n");
        return 1;
    }
    printf("detected %zu hand(s)\n", res.size());
    for (auto& r : res) {
        printf("  box=(%.1f,%.1f,%.1f,%.1f) score=%.3f keypoints=%zu\n",
               r.box.x, r.box.y, r.box.width, r.box.height, r.score, r.keypoints.size());
    }
    return 0;
}
