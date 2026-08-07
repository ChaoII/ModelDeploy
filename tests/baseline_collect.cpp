//
// baseline_collect.cpp — 回归基线收集器（独立 exe）
//
// 用法:
//   baseline_collect.exe --model <path> --image <path> --out <dir> --type <det|obb|seg|pose|cls|face_det|ocr_det|ocr_rec|ocr_cls|pre|raw> [--family <model-family>]
//
// 输出文件: <out>/<model文件名>.<type>.json
//   - 结果类型: { "meta": {...}, "results": <serialize_xxx 数组> }
//   - pre/raw : { "meta": {...}, "tensor": <serialize_tensor 对象> }
//
// pre/raw 模式需要知道加载哪个模型族，优先使用 --family，否则根据模型文件名自动探测。
//

#include <array>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "csrc/vision.h"
#include "baseline_utils.h"

namespace fs = std::filesystem;
namespace vb = modeldeploy::vision::baseline;

using modeldeploy::Tensor;
using modeldeploy::vision::ClassifyResult;
using modeldeploy::vision::DetectionResult;
using modeldeploy::vision::ImageData;
using modeldeploy::vision::InstanceSegResult;
using modeldeploy::vision::KeyPointsResult;
using modeldeploy::vision::LetterBoxRecord;
using modeldeploy::vision::ObbResult;

namespace {

    struct Args {
        std::string model;
        std::string image;
        std::string out;
        std::string type;
        std::string family;
    };

    Args parse_args(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (k == "--model" && i + 1 < argc) a.model = argv[++i];
            else if (k == "--image" && i + 1 < argc) a.image = argv[++i];
            else if (k == "--out" && i + 1 < argc) a.out = argv[++i];
            else if (k == "--type" && i + 1 < argc) a.type = argv[++i];
            else if (k == "--family" && i + 1 < argc) a.family = argv[++i];
        }
        return a;
    }

    void print_usage() {
        std::cerr
            << "usage: baseline_collect --model <path> --image <path> --out <dir>"
            << " --type <det|obb|seg|pose|cls|face_det|ocr_det|ocr_rec|ocr_cls|pre|raw>"
            << " [--family <det|obb|seg|pose|cls|face_det|ocr_det|ocr_rec|ocr_cls>]\n";
    }

    std::string to_lower(std::string s) {
        for (auto& c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return s;
    }

    std::string current_date() {
        std::time_t now = std::time(nullptr);
        std::tm tmv{};
#ifdef _MSC_VER
        localtime_s(&tmv, &now);
#else
        localtime_r(&now, &tmv);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
        return buf;
    }

    // pre/raw 模式下根据模型文件名/路径推断模型族（best-effort，建议显式传 --family）
    std::string auto_detect_family(const std::string& model_file, bool* defaulted = nullptr) {
        if (defaulted) *defaulted = false;
        const std::string path = to_lower(fs::absolute(model_file).string());
        const std::string name = to_lower(fs::path(model_file).filename().string());
        if (path.find("scrfd") != std::string::npos) return "face_det";
        const bool is_ocr = path.find("ocr") != std::string::npos;
        if (is_ocr) {
            if (path.find("rec") != std::string::npos) return "ocr_rec";
            if (path.find("det") != std::string::npos) return "ocr_det";
            if (path.find("cls") != std::string::npos) return "ocr_cls";
        }
        if (name.find("pose") != std::string::npos) return "pose";
        if (name.find("seg") != std::string::npos) return "seg";
        if (name.find("obb") != std::string::npos) return "obb";
        if (name.find("cls") != std::string::npos) return "cls";
        if (defaulted) *defaulted = true;
        return "det";
    }

    std::string find_rec_dict(const std::string& model_file) {
        const std::vector<fs::path> candidates = {
            fs::path(model_file).parent_path() / "ppocrv4_dict.txt",
            fs::path(model_file).parent_path().parent_path() / "ppocrv4_dict.txt",
            fs::path("test_data") / "ppocrv4_dict.txt",
            fs::path("ppocrv4_dict.txt"),
        };
        for (const auto& c : candidates) {
            if (fs::exists(c)) return c.string();
        }
        std::cerr << "warning: ppocrv4_dict.txt not found near model or in test_data/, "
                  << "falling back to test_data/ppocrv4_dict.txt\n";
        return "test_data/ppocrv4_dict.txt";
    }

    modeldeploy::RuntimeOption make_option() {
        modeldeploy::RuntimeOption opt;
        opt.use_cpu();
        opt.set_cpu_thread_num(4);
        return opt;
    }

    json make_meta(const std::string& model_file, const std::string& image_file) {
        return json{
            {"model", fs::path(model_file).filename().string()},
            {"image", fs::path(image_file).filename().string()},
            {"date", current_date()},
            {"backend", "ort-cpu"},
        };
    }

    bool write_json(const std::string& out_dir, const std::string& base,
                    const std::string& type, const json& j, std::string* out_path) {
        fs::path dir(out_dir);
        std::error_code ec;
        fs::create_directories(dir, ec);
        const fs::path out = dir / (base + "." + type + ".json");
        std::ofstream ofs(out);
        if (!ofs) {
            std::cerr << "error: cannot write " << out << std::endl;
            return false;
        }
        ofs << j.dump(2);
        ofs.flush();
        if (!ofs.good()) {
            std::cerr << "error: failed writing " << out << std::endl;
            return false;
        }
        if (out_path) *out_path = out.string();
        return true;
    }

    // 各模型族的前处理分派（run 签名不同，逐个适配）
    bool run_preprocessor(modeldeploy::vision::detection::UltralyticsDet& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        preproc.set_size({640, 640});
        std::vector<LetterBoxRecord> lbs;
        if (!preproc.run(imgs, inputs, &lbs)) {
            std::cerr << "error: UltralyticsDet preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::detection::UltralyticsSeg& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        preproc.set_size({640, 640});
        std::vector<LetterBoxRecord> lbs;
        if (!preproc.run(imgs, inputs, &lbs)) {
            std::cerr << "error: UltralyticsSeg preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::detection::UltralyticsPose& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        preproc.set_size({640, 640});
        std::vector<LetterBoxRecord> lbs;
        if (!preproc.run(imgs, inputs, &lbs)) {
            std::cerr << "error: UltralyticsPose preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::detection::UltralyticsObb& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        preproc.set_size({640, 640});
        std::vector<LetterBoxRecord> lbs;
        if (!preproc.run(imgs, inputs, &lbs)) {
            std::cerr << "error: UltralyticsObb preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::classification::Classification& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        auto images = imgs; // run 接受非 const 指针
        if (!preproc.run(&images, inputs)) {
            std::cerr << "error: Classification preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::face::Scrfd& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        preproc.set_size({640, 640});
        std::vector<LetterBoxRecord> lbs;
        auto images = imgs; // run 接受非 const 指针
        if (!preproc.run(&images, inputs, &lbs)) {
            std::cerr << "error: Scrfd preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::ocr::DBDetector& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        if (!preproc.apply(imgs, inputs)) {
            std::cerr << "error: DBDetector preprocessor.apply failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::ocr::Recognizer& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        std::vector<int> indices{0};
        if (!preproc.run(imgs, inputs, 0, imgs.size(), indices)) {
            std::cerr << "error: Recognizer preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    bool run_preprocessor(modeldeploy::vision::ocr::Classifier& model,
                          const std::vector<ImageData>& imgs, std::vector<Tensor>* inputs) {
        auto& preproc = model.get_preprocessor();
        if (!preproc.run(imgs, inputs, 0, imgs.size())) {
            std::cerr << "error: Classifier preprocessor.run failed" << std::endl;
            return false;
        }
        return true;
    }

    // pre/raw 共用逻辑：写第一个输入张量（pre）或推理后的第一个输出张量（raw）
    template <typename Model>
    bool collect_tensors(Model& model, const std::vector<ImageData>& imgs,
                         const std::string& mode, json* j) {
        std::vector<Tensor> inputs;
        if (!run_preprocessor(model, imgs, &inputs)) {
            return false;
        }
        if (inputs.empty()) {
            std::cerr << "error: preprocessor produced no input tensors" << std::endl;
            return false;
        }
        if (mode == "pre") {
            (*j)["tensor"] = vb::serialize_tensor(inputs[0]);
            return true;
        }
        std::vector<Tensor> outputs;
        if (!model.infer(inputs, &outputs)) {
            std::cerr << "error: model.infer failed" << std::endl;
            return false;
        }
        if (outputs.empty()) {
            std::cerr << "error: infer produced no output tensors" << std::endl;
            return false;
        }
        (*j)["tensor"] = vb::serialize_tensor(outputs[0]);
        return true;
    }

    template <typename Model, typename Result>
    bool collect_yolo(const Args& args, const ImageData& img, json* j, const std::string& mode,
                      json (*ser)(const std::vector<Result>&)) {
        Model model(args.model, make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        std::vector<Result> results;
        if (!model.predict(img, &results, nullptr)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = ser(results);
        return true;
    }

    bool collect_cls(const Args& args, const ImageData& img, json* j, const std::string& mode) {
        modeldeploy::vision::classification::Classification model(args.model, make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        ClassifyResult result;
        if (!model.predict(img, &result)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = vb::serialize_cls(result);
        return true;
    }

    bool collect_face(const Args& args, const ImageData& img, json* j, const std::string& mode) {
        modeldeploy::vision::face::Scrfd model(args.model, make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        std::vector<KeyPointsResult> results;
        if (!model.predict(img, &results, nullptr)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = vb::serialize_pose(results);
        return true;
    }

    bool collect_ocr_det(const Args& args, const ImageData& img, json* j, const std::string& mode) {
        modeldeploy::vision::ocr::DBDetector model(args.model, make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        std::vector<std::array<int, 8>> boxes;
        if (!model.predict(img, &boxes, nullptr)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = vb::serialize_ocr_det(boxes);
        return true;
    }

    bool collect_ocr_rec(const Args& args, const ImageData& img, json* j, const std::string& mode) {
        modeldeploy::vision::ocr::Recognizer model(args.model, find_rec_dict(args.model), make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        std::string text;
        float score = 0;
        if (!model.predict(img, &text, &score, nullptr)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = vb::serialize_ocr_rec(text, score);
        return true;
    }

    bool collect_ocr_cls(const Args& args, const ImageData& img, json* j, const std::string& mode) {
        modeldeploy::vision::ocr::Classifier model(args.model, make_option());
        if (!model.is_initialized()) {
            std::cerr << "error: failed to load model " << args.model << std::endl;
            return false;
        }
        const std::vector<ImageData> imgs{img};
        if (mode == "pre" || mode == "raw") {
            return collect_tensors(model, imgs, mode, j);
        }
        int32_t label = -1;
        float score = 0;
        if (!model.predict(img, &label, &score)) {
            std::cerr << "error: predict failed for " << args.model << std::endl;
            return false;
        }
        (*j)["results"] = vb::serialize_ocr_cls(label, score);
        return true;
    }

    int run(const Args& args) {
        if (args.model.empty() || args.image.empty() || args.out.empty() || args.type.empty()) {
            print_usage();
            return 1;
        }
        const ImageData img = ImageData::imread(args.image);
        if (img.empty()) {
            std::cerr << "error: cannot read image " << args.image << std::endl;
            return 1;
        }

        const std::string& type = args.type;
        const bool is_tensor_mode = (type == "pre" || type == "raw");
        if (!args.family.empty() && !is_tensor_mode) {
            std::cerr << "warning: --family is ignored for --type " << type << "\n";
        }
        std::string family;
        if (is_tensor_mode) {
            if (!args.family.empty()) {
                family = args.family;
            } else {
                bool defaulted = false;
                family = auto_detect_family(args.model, &defaulted);
                if (defaulted) {
                    std::cerr << "auto-detected family: " << family << " (default)\n";
                } else {
                    std::cerr << "auto-detected family: " << family << "\n";
                }
            }
        } else {
            family = type;
        }
        const std::string mode = is_tensor_mode ? type : "result";

        json j;
        j["meta"] = make_meta(args.model, args.image);

        bool ok = false;
        if (family == "det") {
            ok = collect_yolo<modeldeploy::vision::detection::UltralyticsDet, DetectionResult>(
                args, img, &j, mode, vb::serialize_detection);
        } else if (family == "seg") {
            ok = collect_yolo<modeldeploy::vision::detection::UltralyticsSeg, InstanceSegResult>(
                args, img, &j, mode, vb::serialize_seg);
        } else if (family == "pose") {
            ok = collect_yolo<modeldeploy::vision::detection::UltralyticsPose, KeyPointsResult>(
                args, img, &j, mode, vb::serialize_pose);
        } else if (family == "obb") {
            ok = collect_yolo<modeldeploy::vision::detection::UltralyticsObb, ObbResult>(
                args, img, &j, mode, vb::serialize_obb);
        } else if (family == "cls") {
            ok = collect_cls(args, img, &j, mode);
        } else if (family == "face_det") {
            ok = collect_face(args, img, &j, mode);
        } else if (family == "ocr_det") {
            ok = collect_ocr_det(args, img, &j, mode);
        } else if (family == "ocr_rec") {
            ok = collect_ocr_rec(args, img, &j, mode);
        } else if (family == "ocr_cls") {
            ok = collect_ocr_cls(args, img, &j, mode);
        } else {
            std::cerr << "error: unknown type/family: " << family << std::endl;
            return 1;
        }
        if (!ok) {
            return 1;
        }

        std::string out_path;
        if (!write_json(args.out, fs::path(args.model).filename().string(), type, j, &out_path)) {
            return 1;
        }
        std::cout << "written: " << out_path << std::endl;
        return 0;
    }

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
