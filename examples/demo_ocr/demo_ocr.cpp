// demo_ocr: 用 ORT 后端跑 PaddleOCR（det+cls+rec+dict），打印识别文本。
// 用法:
//   demo_ocr <image> [use_gpu 0/1]
#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"

#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image> [use_gpu 0/1]\n", argv[0]);
        return 2;
    }
    const char* kFont = "msyh.ttc";

    const std::string det = "det.onnx";
    const std::string cls = "";
    const std::string rec = "rec.onnx";
    const std::string dict = "dict.txt";
    const std::string img = argv[1];
    bool use_gpu = false;
    if (argc >= 3) {
        const std::string a = argv[2];
        use_gpu = (a == "1" || a == "true" || a == "on" || a == "gpu");
    }

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    if (use_gpu) {
        opt.set_device(modeldeploy::Device::GPU, 0);
    } else {
        opt.set_device(modeldeploy::Device::CPU);
        opt.set_cpu_thread_num(4);
    }

    auto m = std::make_unique<modeldeploy::vision::ocr::PaddleOCR>(det, cls, rec, dict, opt);
    if (!m->is_initialized()) {
        std::fprintf(stderr, "init failed\n");
        return 1;
    }
    m->set_rec_batch_size(1);
    m->get_detector()->get_preprocessor().set_max_side_len(960);

    auto im = modeldeploy::vision::ImageData::imread(img);
    if (im.empty()) {
        std::fprintf(stderr, "cannot read image: %s\n", img.c_str());
        return 1;
    }

    modeldeploy::vision::OCRResult res;
    if (!m->predict(im, &res, nullptr)) {
        std::fprintf(stderr, "predict failed\n");
        return 1;
    }
    auto vis = modeldeploy::vision::vis_ocr(im, res, kFont, 24);
    (void)vis.imwrite("train_ocr.jpg");
    // vis.imshow("train_ocr");
    modeldeploy::vision::dis_ocr(res);
    return 0;
}
