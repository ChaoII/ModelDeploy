//
// ServingServer 内置通用 HandleBuilder 实现。
//
#include "serving/builtin_builder.h"

#include "serving/manifest.h"

#if defined(BUILD_VISION)
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "serving/adapters.h"
#include "serving/model_entry.h"

#include "runtime/runtime_option.h"
#include "vision/common/result_json.h"
#include "vision/common/visualize/visualize.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/classification/classification.h"
#include "vision/iseg/ultralytics_seg.h"
#include "vision/pose/ultralytics_pose.h"
#include "vision/obb/ultralytics_obb.h"
#include "vision/sem/ultralytics_sem.h"
#include "vision/depth/ultralytics_depth.h"
#include "vision/ocr/ppocr.h"
#include "vision/face/face_det/scrfd.h"
#include "vision/lpr/lpr_pipeline/lpr_pipeline.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace modeldeploy::serving {
namespace {

// 仅含目录元数据的句柄（无 infer）：构造失败或未知 type 时返回，由 load 置 Failed。
ModelHandle meta_handle(const ManifestModel& m) {
    ModelHandle h;
    h.name = m.id;
    h.display = m.display;
    h.version = "1";
    h.type = m.type;
    h.input_size = m.input_size;
    h.labels = m.labels;
    return h;
}

// 构造成功：make_model_handle 的核心句柄已带 infer，回填元数据与可选 vis。
template <typename M>
ModelHandle built_handle(
    const ManifestModel& m, std::unique_ptr<M> model,
    std::function<ImageData(ImageData&, const typename M::result_type&)> vis = {}) {
    ModelHandle h = make_model_handle<M>(m.id, std::move(model), {}, std::move(vis));
    h.display = m.display;
    h.type = m.type;
    h.input_size = m.input_size;
    h.labels = m.labels;
    return h;
}

// 从模型尽量取名称标签；取不到返回空（vis 用默认配色，不影响推理）。
std::unordered_map<int, std::string> model_labels(
    const std::function<std::unordered_map<int, std::string>()>& fn) {
    try {
        return fn();
    } catch (...) {
        return {};
    }
}

}  // namespace

HandleBuilder make_builtin_builder(const ServingConfig& cfg) {
    const std::string font = cfg.font_path;
    return [font](const ManifestModel& m, const std::string&) -> ModelHandle {
        ModelHandle meta = meta_handle(m);
        RuntimeOption opt;
        opt.use_ort_backend();
        opt.set_device(Device::CPU, 0);  // CPU 起步；有 GPU/权重时改 Device::GPU

        if (m.type == "det") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsDet,
                                       std::vector<DetectionResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto lm = model_labels([&] { return model->get_label_map("names"); });
                auto vis = [lm, font](ImageData& im, const std::vector<DetectionResult>& r) {
                    return vis_det(im, r, 0.5, lm, font, 12, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "cls") {
            try {
                using MM = ClassifyAdapter<vision::ClassifyResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [font](ImageData& im, const vision::ClassifyResult& r) {
                    return vis_cls(im, r, 1, 0.5, font, 12, 0.15, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "seg") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsSeg,
                                       std::vector<InstanceSegResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [font](ImageData& im, const std::vector<InstanceSegResult>& r) {
                    return vis_iseg(im, r, 0.5, font, 12, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "pose") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsPose,
                                       std::vector<KeyPointsResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [font](ImageData& im, const std::vector<KeyPointsResult>& r) {
                    return vis_pose(im, r, font, 12, 4, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "obb") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsObb,
                                       std::vector<ObbResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [font](ImageData& im, const std::vector<ObbResult>& r) {
                    return vis_obb(im, r, 0.5, font, 12, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "sem") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsSem, vision::SemSegResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto lm = model_labels([&] { return model->get_label_map("names"); });
                auto vis = [lm](ImageData& im, const SemSegResult& r) {
                    return vis_sem(im, r, lm, 0.5, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "depth") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsDepth, vision::DepthResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [](ImageData& im, const DepthResult& r) {
                    return vis_depth(im, r, true, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "face") {
            try {
                using MM = ResultModel<vision::face::Scrfd, std::vector<KeyPointsResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                model->get_preprocessor().set_size(m.input_size);
                auto vis = [font](ImageData& im, const std::vector<KeyPointsResult>& r) {
                    return vis_keypoints(im, r, font, 12, 4, 0.3, false, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "ocr") {
            try {
                // PaddleOCR 构造收 (det, cls, rec, dict, option)。
                using MM = ResultModel<vision::ocr::PaddleOCR, vision::OCRResult>;
                if (m.rec_f.empty() || m.dict_f.empty()) return meta;
                auto model = std::make_unique<MM>(m.model_f, m.cls_f, m.rec_f, m.dict_f, opt);
                if (!model->is_initialized()) return meta;
                auto vis = [font](ImageData& im, const OCRResult& r) {
                    return vis_ocr(im, r, font, 12, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        if (m.type == "lpr") {
            try {
                // LprPipeline 构造收 (det, rec, option)。
                using MM = LprAdapter<std::vector<vision::LprResult>>;
                if (m.rec_f.empty()) return meta;
                auto model = std::make_unique<MM>(m.model_f, m.rec_f, opt);
                if (!model->is_initialized()) return meta;
                auto vis = [font](ImageData& im, const std::vector<LprResult>& r) {
                    return vis_lpr(im, r, font, 12, 4, 0.3, false);
                };
                return built_handle(m, std::move(model), std::move(vis));
            } catch (...) { return meta; }
        }
        return meta;  // 未知 type：目录内可列，load 置 Failed
    };
}

}  // namespace modeldeploy::serving

#else  // !BUILD_VISION

namespace modeldeploy::serving {

HandleBuilder make_builtin_builder(const ServingConfig&) {
    return [](const ManifestModel&, const std::string&) -> ModelHandle {
        ModelHandle h;
        h.error = "builtin builder requires BUILD_VISION; inject a custom HandleBuilder";
        return h;
    };
}

}  // namespace modeldeploy::serving

#endif  // BUILD_VISION
