#include "inference_engine.hpp"
#include "runtime_factory.hpp"
#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/common/result.h"
#include "csrc/vision/common/struct.h"
#include "csrc/vision/utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <limits>

using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace {
// 旋转框 → 轴对齐外接矩形（事件契约只有 x/y/w/h，旋转框降维为外接 bbox）。
// angle 为 OpenCV RotatedRect 约定（角度制），按中心/宽高/角度取 4 角点后求包围盒。
void rotated_rect_bounds(const RotatedRect& rr, float* xmin, float* ymin,
                         float* xmax, float* ymax) {
    const float rad = rr.angle * 3.14159265358979323846f / 180.0f;
    const float c = std::cos(rad), s = std::sin(rad);
    const float hw = rr.width * 0.5f, hh = rr.height * 0.5f;
    const float dx[4] = {hw, hw, -hw, -hw};
    const float dy[4] = {hh, -hh, -hh, hh};
    float x0 = rr.xc, y0 = rr.yc, x1 = rr.xc, y1 = rr.yc;
    for (int i = 0; i < 4; ++i) {
        const float px = rr.xc + dx[i] * c - dy[i] * s;
        const float py = rr.yc + dx[i] * s + dy[i] * c;
        x0 = std::min(x0, px); x1 = std::max(x1, px);
        y0 = std::min(y0, py); y1 = std::max(y1, py);
    }
    *xmin = x0; *ymin = y0; *xmax = x1; *ymax = y1;
}

// NV12（host/device）→ packed BGR（CPU）：人脸属性/活体等 CPU 融合预处只支持 BGR，
// 直接按单通道读 NV12 会越界（与 classification/scrfd 的同类崩溃修复一致）。
ImageData to_packed_bgr(const ImageData& image) {
    if (image.empty() || image.type() != MdImageType::NV12 || image.plane_count() < 2) return image;
    if (image.device() != Device::CPU) {
        ImageData cpu;
        if (!image.toCpu(&cpu)) return {};
        return ImageData::cvt_color(cpu, ColorConvertType::CVT_NV122PKG_BGR);
    }
    return ImageData::cvt_color(image, ColorConvertType::CVT_NV122PKG_BGR);
}

// face_attr 的“年龄年轻”分数：年龄分布中下标 <= 35 的概率质量 P(age<=35)（模型已 softmax）。
constexpr int kAgeYoungMax = 35;

float age_young_score(const std::vector<float>& dist) {
    float s = 0.f;
    for (size_t i = 0; i < dist.size() && i <= static_cast<size_t>(kAgeYoungMax); ++i) s += dist[i];
    return std::clamp(s, 0.f, 1.f);
}

// 深度图最近有效距离（米）：忽略非有限/非正值，取最小值；全无效返回 false（fail-closed）。
bool nearest_depth(const std::vector<float>& depth, float* out) {
    float best = std::numeric_limits<float>::infinity();
    for (float v : depth) {
        if (!std::isfinite(v) || v <= 0.f) continue;
        if (v < best) best = v;
    }
    if (!std::isfinite(best) || !out) return false;
    *out = best;
    return true;
}

// 类别名：优先配置 labels，缺失/越界回退 "class<id>"（保证事件 label 非空）。
std::string class_name_of(const ModelConfig& cfg, int id) {
    if (id >= 0 && static_cast<size_t>(id) < cfg.labels.size() && !cfg.labels[id].empty())
        return cfg.labels[id];
    return "class" + std::to_string(id);
}

// 分割掩码是否命中像素点。掩码为「框局部坐标」（shape={h,w}，原点 = 框左上角，
// 见 iseg/sam postprocessor），故先把像素点换算到框内下标再取布尔值。
bool mask_hits_point(const InstanceSegResult& r, float px, float py) {
    if (r.mask.shape.size() < 2) return false;
    const int h = static_cast<int>(r.mask.shape[0]);
    const int w = static_cast<int>(r.mask.shape[1]);
    if (h <= 0 || w <= 0 || r.mask.buffer.size() < static_cast<size_t>(h) * static_cast<size_t>(w))
        return false;
    const int mx = static_cast<int>(px - r.box.x);
    const int my = static_cast<int>(py - r.box.y);
    if (mx < 0 || my < 0 || mx >= w || my >= h) return false;
    return r.mask.buffer[static_cast<size_t>(my) * static_cast<size_t>(w) + mx] != 0;
}

// 交互式提示点 → 选中的分割实例（返回下标，-1 = 无可用实例）：
// 1) 有提示点时，取「掩码命中该点」的实例中分数最高者；
// 2) 无提示点、或提示点未命中任何掩码（含掩码缺失）→ 回退全体最高分实例；
// 3) 无几何（宽高非正）的实例不参与。
int select_segment_index(const std::vector<InstanceSegResult>& segs, const float* point) {
    int best_hit = -1;
    float best_hit_score = -1.f;
    int best_any = -1;
    float best_any_score = -1.f;
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        if (!(s.box.width > 0.f && s.box.height > 0.f)) continue;
        if (s.score > best_any_score) {
            best_any_score = s.score;
            best_any = static_cast<int>(i);
        }
        if (point && mask_hits_point(s, point[0], point[1]) && s.score > best_hit_score) {
            best_hit_score = s.score;
            best_hit = static_cast<int>(i);
        }
    }
    return best_hit >= 0 ? best_hit : best_any;
}

// 提示点（归一化 [0,1]）→ 像素坐标；未配置或帧尺寸非法返回 false（调用方走最高分兜底）。
bool prompt_to_pixel(const ModelConfig& cfg, float frame_w, float frame_h, float* out) {
    if (!out || cfg.prompt_point.size() < 2) return false;
    if (!(frame_w > 0.f && frame_h > 0.f)) return false;
    out[0] = std::clamp(cfg.prompt_point[0], 0.f, 1.f) * frame_w;
    out[1] = std::clamp(cfg.prompt_point[1], 0.f, 1.f) * frame_h;
    return true;
}

// 原地 L2 归一化：范数为 0/非有限时返回 false（不产生无效嵌入，fail-closed）。
bool l2_normalize(std::vector<float>* v) {
    if (!v || v->empty()) return false;
    double sum = 0.0;
    for (float x : *v) sum += static_cast<double>(x) * static_cast<double>(x);
    const double norm = std::sqrt(sum);
    if (!std::isfinite(norm) || !(norm > 1e-12)) return false;
    for (float& x : *v) x = static_cast<float>(static_cast<double>(x) / norm);
    return true;
}
}  // namespace

bool InferenceEngine::load(const ModelConfig& cfg) {
    if (loaded_) unload();
    cfg_ = cfg;

    RuntimeOption opt = build_runtime_option(cfg);

    try {
        if (cfg.type == "detection") {
            det_model_ = std::make_unique<detection::UltralyticsDet>(cfg.path, opt);
            if (!det_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Detection model init failed: " << cfg.path << std::endl;
                det_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                det_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "face_detection") {
            face_model_ = std::make_unique<face::Scrfd>(cfg.path, opt);
            if (!face_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Face model init failed: " << cfg.path << std::endl;
                face_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                face_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "classification") {
            cls_model_ = std::make_unique<classification::Classification>(cfg.path, opt);
            if (!cls_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Classification model init failed: " << cfg.path << std::endl;
                cls_model_.reset();
                return false;
            }
        } else if (cfg.type == "obb") {
            // 旋转目标检测（YOLO-OBB）：SDK 输出旋转框，事件侧降维为轴对齐外接 bbox
            obb_model_ = std::make_unique<detection::UltralyticsObb>(cfg.path, opt);
            if (!obb_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] OBB model init failed: " << cfg.path << std::endl;
                obb_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                obb_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "iseg") {
            // 实例分割（YOLO-Seg）：仅把实例框送入事件（掩码不入事件，见批次文档）
            iseg_model_ = std::make_unique<detection::UltralyticsSeg>(cfg.path, opt);
            if (!iseg_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] ISeg model init failed: " << cfg.path << std::endl;
                iseg_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                iseg_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "pose") {
            // 人体姿态关键点（YOLO-Pose）：每人框 + 17 点 (x, y, score)，关键点写入事件
            pose_model_ = std::make_unique<detection::UltralyticsPose>(cfg.path, opt);
            if (!pose_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Pose model init failed: " << cfg.path << std::endl;
                pose_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                pose_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "face_attr") {
            // 人脸性别/年龄属性（SCRFD 检测 + 性别/年龄分类，输出概率分数）
            // 路径口径：path=SCRFD、cls_path=性别、rec_path=年龄（见 config_adapter 归一化）
            if (cfg.cls_path.empty() || cfg.rec_path.empty()) {
                std::cerr << "[InferenceEngine] face_attr requires cls_path(gender) and rec_path(age)"
                          << std::endl;
                return false;
            }
            fa_det_model_ = std::make_unique<face::Scrfd>(cfg.path, opt);
            fa_gender_model_ = std::make_unique<face::SeetaFaceGender>(cfg.cls_path, opt);
            fa_age_model_ = std::make_unique<face::SeetaFaceAge>(cfg.rec_path, opt);
            if (!fa_det_model_->is_initialized() || !fa_gender_model_->is_initialized() ||
                !fa_age_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] face_attr init failed: " << cfg.path << " | "
                          << cfg.cls_path << " | " << cfg.rec_path << std::endl;
                fa_det_model_.reset();
                fa_gender_model_.reset();
                fa_age_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                fa_det_model_->get_preprocessor().set_size(cfg_.input_size);
            fa_det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        } else if (cfg.type == "face_as") {
            // 人脸活体（SCRFD 检测 + fas_first 活体分数；fas_second 可选，用作整体伪造门控）
            if (cfg.cls_path.empty()) {
                std::cerr << "[InferenceEngine] face_as requires cls_path(fas_first)" << std::endl;
                return false;
            }
            fas_det_model_ = std::make_unique<face::Scrfd>(cfg.path, opt);
            fas_first_model_ = std::make_unique<face::SeetaFaceAsFirst>(cfg.cls_path, opt);
            if (!cfg.rec_path.empty())
                fas_second_model_ = std::make_unique<face::SeetaFaceAsSecond>(cfg.rec_path, opt);
            if (!fas_det_model_->is_initialized() || !fas_first_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] face_as init failed: " << cfg.path << " | "
                          << cfg.cls_path << std::endl;
                fas_det_model_.reset();
                fas_first_model_.reset();
                fas_second_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                fas_det_model_->get_preprocessor().set_size(cfg_.input_size);
            fas_det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        } else if (cfg.type == "depth") {
            // 单目深度估计（yolo26n-depth）：输出整帧最近距离（见 infer_depth）
            depth_model_ = std::make_unique<detection::UltralyticsDepth>(cfg.path, opt);
            if (!depth_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Depth model init failed: " << cfg.path << std::endl;
                depth_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                depth_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "sem") {
            // 语义分割（yolo26n-sem，cityscapes 19 类）：整帧占比写入 attributes（见 infer_sem）
            sem_model_ = std::make_unique<detection::UltralyticsSem>(cfg.path, opt);
            if (!sem_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] Sem model init failed: " << cfg.path << std::endl;
                sem_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                sem_model_->get_preprocessor().set_size(cfg_.input_size);
        } else if (cfg.type == "face_landmark") {
            // 人脸关键点（InsightFace 2d106det，106 点）：输入为人脸裁剪图，输出 keypoints 事件字段
            face_landmark_model_ = std::make_unique<landmark::FaceLandmark>(cfg.path, opt);
            if (!face_landmark_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] FaceLandmark model init failed: " << cfg.path
                          << std::endl;
                face_landmark_model_.reset();
                return false;
            }
        } else if (cfg.type == "doc") {
            // 文档/表格：与 OCR 同口径（det 主模型 + cls 方向分类 + rec 识别 + 字典），
            // 文本写入 objects[].text（版面/表格结构叶子云端未实现，本族不伪造结构）
            ocr_model_ = std::make_unique<ocr::PaddleOCR>(
                cfg.path, cfg.cls_path, cfg.rec_path, cfg.dict_path, opt);
            if (!ocr_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] DOC(OCR) pipeline init failed" << std::endl;
                ocr_model_.reset();
                return false;
            }
            ocr_model_->set_rec_batch_size(8);
            ocr_model_->get_detector()->get_preprocessor().set_max_side_len(1440);
        } else if (cfg.type == "barcode") {
            // 条码/二维码：原生 ZXing 解码（无 DNN 权重，path 可空）
            barcode_detector_ = std::make_unique<barcode::BarcodeDetector>();
        } else if (cfg.type == "sam") {
            // 交互式分割（FastSAM-s，1×3×640×640 → [1,37,8400] box + [1,32,160,160] 掩码原型）：
            // 一次出 box + mask，按 prompt_point 选段（见 infer_sam）
            sam_model_ = std::make_unique<seg::FastSam>(cfg.path, opt);
            if (!sam_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] FastSAM model init failed: " << cfg.path << std::endl;
                sam_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                sam_model_->get_preprocessor().set_size(cfg_.input_size);
            sam_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        } else if (cfg.type == "face_rec") {
            // 人脸识别复合模型：SCRFD 检测 + ArcFace 嵌入（w600k_r50，112→512-d，见 infer_face_rec）
            if (cfg.rec_path.empty()) {
                std::cerr << "[InferenceEngine] face_rec requires rec_path(embedding)" << std::endl;
                return false;
            }
            fr_det_model_ = std::make_unique<face::Scrfd>(cfg.path, opt);
            fr_rec_model_ = std::make_unique<face::InsightFaceRecognition>(cfg.rec_path, opt);
            if (!fr_det_model_->is_initialized() || !fr_rec_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] face_rec init failed: " << cfg.path << " | "
                          << cfg.rec_path << std::endl;
                fr_det_model_.reset();
                fr_rec_model_.reset();
                return false;
            }
            if (cfg_.input_size.size() == 2)
                fr_det_model_->get_preprocessor().set_size(cfg_.input_size);
            fr_det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        } else if (cfg.type == "pedestrian_attribute") {
            pa_model_ = std::make_unique<modeldeploy::vision::pipeline::PedestrianAttribute>(
                cfg.path, cfg.rec_path, opt);
            if (!pa_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] pedestrian_attribute init failed" << std::endl;
                pa_model_.reset();
                return false;
            }
            pa_model_->set_det_threshold(cfg_.confidence_threshold);
            // 从模型元数据读取输入尺寸（NCHW -> {W,H}），优先于配置默认值，避免尺寸不匹配
            try {
                const auto det_shape = pa_model_->get_detector()->get_input_info(0).shape;
                if (det_shape.size() >= 4 && det_shape[2] > 0 && det_shape[3] > 0)
                    pa_model_->set_det_input_size({static_cast<int>(det_shape[3]), static_cast<int>(det_shape[2])});
                else if (cfg_.input_size.size() == 2)
                    pa_model_->set_det_input_size(cfg_.input_size);
            } catch (...) {
                if (cfg_.input_size.size() == 2) pa_model_->set_det_input_size(cfg_.input_size);
            }
            try {
                const auto cls_shape = pa_model_->get_classifier()->get_input_info(0).shape;
                if (cls_shape.size() >= 4 && cls_shape[2] > 0 && cls_shape[3] > 0)
                    pa_model_->set_cls_input_size({static_cast<int>(cls_shape[3]), static_cast<int>(cls_shape[2])});
                else
                    pa_model_->set_cls_input_size({192, 256});
            } catch (...) {
                pa_model_->set_cls_input_size({192, 256});
            }
            pa_model_->set_cls_batch_size(8);
        } else if (cfg.type == "ocr") {
            // OCR pipeline：det 主模型 + cls 方向分类 + rec 识别 + 字典
            ocr_model_ = std::make_unique<ocr::PaddleOCR>(
                cfg.path, cfg.cls_path, cfg.rec_path, cfg.dict_path, opt);
            if (!ocr_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] OCR pipeline init failed" << std::endl;
                ocr_model_.reset();
                return false;
            }
            ocr_model_->set_rec_batch_size(8);
            // det 输入长边上限 1440，兼顾大图精度与速度
            ocr_model_->get_detector()->get_preprocessor().set_max_side_len(1440);
        } else if (cfg.type == "lpr") {
            // LPR 车牌 pipeline：det 定位 + rec 识别（车牌号 + 颜色）
            lpr_model_ = std::make_unique<lpr::LprPipeline>(cfg.path, cfg.rec_path, opt);
            if (!lpr_model_->is_initialized()) {
                std::cerr << "[InferenceEngine] LPR pipeline init failed: "
                          << cfg.path << " | " << cfg.rec_path << std::endl;
                lpr_model_.reset();
                return false;
            }
        } else {
            std::cerr << "[InferenceEngine] Unsupported model type: " << cfg.type << std::endl;
            return false;
        }

        // 将配置中的置信度阈值传递到模型后处理
        if (det_model_) {
            det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
        if (obb_model_) {
            obb_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
        if (iseg_model_) {
            iseg_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
        if (pose_model_) {
            pose_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
        if (face_model_) {
            face_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
        }
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Exception loading model: " << e.what() << std::endl;
        return false;
    }

    loaded_ = true;
    std::cout << "[InferenceEngine] Loaded " << cfg.name << " [" << cfg.type
              << "] " << cfg.path << " backend=" << cfg.backend << std::endl;
    return true;
}

std::string InferenceEngine::make_cache_key(const ModelConfig& cfg) {
    return cfg.path + "|" + cfg.backend + "|" + cfg.device + "|" + cfg.type
           + "|" + std::to_string(cfg.input_size[0]) + "x" + std::to_string(cfg.input_size[1]);
}

bool InferenceEngine::clone_detection_from(
    const detection::UltralyticsDet& proto, const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;

    // clone() = 新 instance + 共享 Runtime (含 ORT Session)
    auto cloned = proto.clone();
    if (!cloned || !cloned->is_initialized()) {
        std::cerr << "[InferenceEngine] Failed to clone detection model" << std::endl;
        return false;
    }
    det_model_ = std::move(cloned);
    if (cfg_.input_size.size() == 2)
        det_model_->get_preprocessor().set_size(cfg_.input_size);
    det_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);

    loaded_ = true;
    std::cout << "[InferenceEngine] Cloned detection: " << cfg.name
              << " (shared ORT session)" << std::endl;
    return true;
}

void InferenceEngine::adopt_face_model(
    std::unique_ptr<face::Scrfd> model, const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;
    face_model_ = std::move(model);
    face_model_->get_postprocessor().set_conf_threshold(cfg_.confidence_threshold);
    loaded_ = true;
    std::cout << "[InferenceEngine] Adopted face: " << cfg.name
              << " (shared ORT session)" << std::endl;
}

void InferenceEngine::set_shared_detector(std::shared_ptr<BatchedDetector> det,
                                          const ModelConfig& cfg) {
    if (loaded_) this->unload();
    cfg_ = cfg;
    shared_det_ = std::move(det);
    loaded_ = shared_det_ && shared_det_->ok();
    if (!loaded_) {
        std::cerr << "[InferenceEngine] shared detector unavailable: "
                  << (shared_det_ ? shared_det_->error() : "null") << std::endl;
    }
}

bool InferenceEngine::predict_detection(const ImageData& image,
                                        std::vector<DetectionResult>* out) {
    if (shared_det_) return shared_det_->predict(image, out);
    if (!det_model_) return false;
    return det_model_->predict(image, out);
}

void InferenceEngine::unload() {
    det_model_.reset();
    cls_model_.reset();
    obb_model_.reset();
    iseg_model_.reset();
    pose_model_.reset();
    face_model_.reset();
    fa_det_model_.reset();
    fa_gender_model_.reset();
    fa_age_model_.reset();
    fas_det_model_.reset();
    fas_first_model_.reset();
    fas_second_model_.reset();
    depth_model_.reset();
    sem_model_.reset();
    face_landmark_model_.reset();
    barcode_detector_.reset();
    sam_model_.reset();
    fr_det_model_.reset();
    fr_rec_model_.reset();
    pa_model_.reset();
    ocr_model_.reset();
    lpr_model_.reset();
    shared_det_.reset();
    loaded_ = false;
}

bool InferenceEngine::infer(const ImageData& image, InferResult* result) {
    if (!loaded_ || !result) return false;
    result->model_name = cfg_.name;
    result->type = cfg_.type;

    if (cfg_.type == "detection")
        return infer_detection(image, result);
    if (cfg_.type == "face_detection")
        return infer_face(image, result);
    if (cfg_.type == "classification")
        return infer_classification(image, result);
    if (cfg_.type == "obb")
        return infer_obb(image, result);
    if (cfg_.type == "iseg")
        return infer_iseg(image, result);
    if (cfg_.type == "pose")
        return infer_pose(image, result);
    if (cfg_.type == "face_attr")
        return infer_face_attr(image, result);
    if (cfg_.type == "face_as")
        return infer_face_as(image, result);
    if (cfg_.type == "depth")
        return infer_depth(image, result);
    if (cfg_.type == "sem")
        return infer_sem(image, result);
    if (cfg_.type == "face_landmark")
        return infer_face_landmark(image, result);
    if (cfg_.type == "doc")
        return infer_doc(image, result);
    if (cfg_.type == "barcode")
        return infer_barcode(image, result);
    if (cfg_.type == "sam")
        return infer_sam(image, result);
    if (cfg_.type == "face_rec")
        return infer_face_rec(image, result);
    if (cfg_.type == "pedestrian_attribute")
        return infer_pedestrian_attribute(image, result);
    if (cfg_.type == "ocr")
        return infer_ocr(image, result);
    if (cfg_.type == "lpr")
        return infer_lpr(image, result);
    return false;
}

bool InferenceEngine::infer_detection(const ImageData& image, InferResult* result) {
    if (!shared_det_ && !det_model_) return false;
    std::vector<DetectionResult> det_results;
    if (!predict_detection(image, &det_results)) return false;
    for (auto& d : det_results) {
        DetectionBox box;
        box.x = d.box.x; box.y = d.box.y;
        box.w = d.box.width; box.h = d.box.height;
        box.score = d.score;
        box.label_id = d.label_id;
        result->boxes.push_back(box);
    }
    return true;
}

bool InferenceEngine::infer_classification(const ImageData& image, InferResult* result) {
    if (!cls_model_ || !result) return false;
    // 分类预处理只接受 packed BGR；NV12（硬解/摄像头帧）必须显式转 BGR。
    // 否则 CPU fused 预处理会把单通道 Y 平面当 3 通道读，造成越界（回归修复）。
    ImageData input = image;
    if (image.type() == MdImageType::NV12 && image.plane_count() >= 2) {
        if (image.device() != modeldeploy::Device::CPU) {
            ImageData cpu;
            if (!image.toCpu(&cpu)) return false;
            input = ImageData::cvt_color(cpu, ColorConvertType::CVT_NV122PKG_BGR);
        } else {
            input = ImageData::cvt_color(image, ColorConvertType::CVT_NV122PKG_BGR);
        }
        if (input.empty()) return false;
    }
    ClassifyResult cls_result;
    if (!cls_model_->predict(input, &cls_result)) return false;
    // 分类无几何：为让云端 object_present/attribute 叶子可匹配（无需云改动），产出整帧框
    // （像素 0,0,W,H → 事件归一化后 0,0,1,1），label = top-1 类别名；attributes 存该类别分数，
    // 使 attribute 叶子（attributes[field]）也具备可比较的数据。top_k 默认 1，通常仅一个框。
    const float W = static_cast<float>(image.width());
    const float H = static_cast<float>(image.height());
    for (size_t i = 0; i < cls_result.label_ids.size(); ++i) {
        const int id = cls_result.label_ids[i];
        const float score = i < cls_result.scores.size() ? cls_result.scores[i] : 0.f;
        const std::string label =
            (id >= 0 && static_cast<size_t>(id) < cfg_.labels.size())
                ? cfg_.labels[static_cast<size_t>(id)]
                : std::to_string(id);
        DetectionBox box;
        box.x = 0.f; box.y = 0.f; box.w = W; box.h = H;
        box.label_id = id;
        box.score = score;
        box.label_name = label;
        box.attributes.emplace_back(label, score);
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_obb(const ImageData& image, InferResult* result) {
    if (!obb_model_ || !result) return false;
    std::vector<ObbResult> obb_results;
    if (!obb_model_->predict(image, &obb_results)) return false;
    for (const auto& o : obb_results) {
        // 旋转框 → 轴对齐外接矩形：复用既有 bbox 事件契约（保持契约不变）
        float xmin = 0.f, ymin = 0.f, xmax = 0.f, ymax = 0.f;
        rotated_rect_bounds(o.rotated_box, &xmin, &ymin, &xmax, &ymax);
        DetectionBox box;
        box.x = xmin; box.y = ymin; box.w = xmax - xmin; box.h = ymax - ymin;
        box.score = o.score;
        box.label_id = o.label_id;
        box.label_name = (o.label_id >= 0 && static_cast<size_t>(o.label_id) < cfg_.labels.size())
                             ? cfg_.labels[static_cast<size_t>(o.label_id)]
                             : std::to_string(o.label_id);
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_iseg(const ImageData& image, InferResult* result) {
    if (!iseg_model_ || !result) return false;
    std::vector<InstanceSegResult> seg_results;
    if (!iseg_model_->predict(image, &seg_results)) return false;
    // SAM_SEG 兼容：云端场景目录 SAM_SEG 的 pipeline 声明为 [iseg]（type="iseg"），
    // 若任务带提示点则按交互式分割语义只输出「命中提示点（无命中则最高分）」的单个 segment
    // 对象（label="segment"）；未配置 prompt_point 时行为与既有实例分割完全一致。
    float pt[2] = {0.f, 0.f};
    if (prompt_to_pixel(cfg_, static_cast<float>(image.width()),
                        static_cast<float>(image.height()), pt)) {
        const int idx = select_segment_index(seg_results, pt);
        if (idx < 0) return true;
        const auto& s = seg_results[static_cast<size_t>(idx)];
        DetectionBox box;
        box.x = s.box.x; box.y = s.box.y; box.w = s.box.width; box.h = s.box.height;
        box.score = s.score;
        box.label_id = s.label_id;
        box.label_name = "segment";
        result->boxes.push_back(std::move(box));
        return true;
    }
    for (const auto& s : seg_results) {
        // 掩码不入事件（带宽/契约），只送实例框；语义见批次文档
        DetectionBox box;
        box.x = s.box.x; box.y = s.box.y; box.w = s.box.width; box.h = s.box.height;
        box.score = s.score;
        box.label_id = s.label_id;
        box.label_name = (s.label_id >= 0 && static_cast<size_t>(s.label_id) < cfg_.labels.size())
                             ? cfg_.labels[static_cast<size_t>(s.label_id)]
                             : std::to_string(s.label_id);
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_pose(const ImageData& image, InferResult* result) {
    if (!pose_model_ || !result) return false;
    std::vector<KeyPointsResult> pose_results;
    if (!pose_model_->predict(image, &pose_results)) return false;
    for (const auto& p : pose_results) {
        // 每人一个人体框（像素坐标），label 取模型标签（缺省 "person"），
        // 关键点 [x, y, score] 与框同坐标系（像素），由 EventBus 归一化后写入事件
        DetectionBox box;
        box.x = p.box.x; box.y = p.box.y;
        box.w = p.box.width; box.h = p.box.height;
        box.score = p.score;
        box.label_id = p.label_id;
        box.label_name = (p.label_id >= 0 && static_cast<size_t>(p.label_id) < cfg_.labels.size())
                             ? cfg_.labels[static_cast<size_t>(p.label_id)]
                             : "person";
        box.keypoints.reserve(p.keypoints.size());
        for (const auto& k : p.keypoints) {
            box.keypoints.push_back({k.x, k.y, k.z});
        }
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_face(const ImageData& image, InferResult* result) {
    if (!face_model_) return false;
    std::vector<KeyPointsResult> face_results;
    if (!face_model_->predict(image, &face_results)) return false;
    float thresh = cfg_.confidence_threshold;
    for (auto& f : face_results) {
        if (f.score < thresh) continue;
        DetectionBox box;
        box.x = f.box.x; box.y = f.box.y;
        box.w = f.box.width; box.h = f.box.height;
        box.score = f.score;
        box.label_id = f.label_id;
        result->boxes.push_back(box);
        std::vector<FaceKeypoint> kps;
        kps.reserve(f.keypoints.size());
        for (const auto& p : f.keypoints) {
            FaceKeypoint kp;
            kp.x = p.x;
            kp.y = p.y;
            kps.push_back(kp);
        }
        result->keypoints.push_back(std::move(kps));
    }
    return true;
}

bool InferenceEngine::infer_face_attr(const ImageData& image, InferResult* result) {
    if (!fa_det_model_ || !result) return false;
    // CPU 融合预处只支持 packed BGR：NV12 帧先转换（同 classification/scrfd 崩溃修复）
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    std::vector<KeyPointsResult> faces;
    if (!fa_det_model_->predict(frame, &faces)) return false;
    const float thresh = cfg_.confidence_threshold;
    // 默认属性名与云端 FACE_ATTR 契约一致（gender_male/age_young）；配置 attributes 可按序覆盖
    // （复用 pedestrian_attribute 的「下标 → 属性名」机制）。
    static const std::vector<std::string> kDefaultNames = {"gender_male", "age_young"};
    const std::vector<std::string>& names =
        cfg_.attributes.empty() ? kDefaultNames : cfg_.attributes;
    const float W = static_cast<float>(frame.width());
    const float H = static_cast<float>(frame.height());
    for (const auto& f : faces) {
        if (f.score < thresh) continue;
        // 人脸区域外扩 10%，给属性模型留出上下文
        const float mx = f.box.width * 0.1f, my = f.box.height * 0.1f;
        const float x1 = std::max(0.f, f.box.x - mx), y1 = std::max(0.f, f.box.y - my);
        const float x2 = std::min(W, f.box.x + f.box.width + mx);
        const float y2 = std::min(H, f.box.y + f.box.height + my);
        if (x2 - x1 < 2.f || y2 - y1 < 2.f) continue;
        ImageData face = frame.crop({x1, y1, x2 - x1, y2 - y1});
        if (face.empty()) continue;
        std::vector<float> gender_probs;
        if (!fa_gender_model_->predict_probs(face, &gender_probs)) continue;
        std::vector<float> age_dist;
        const bool got_age = fa_age_model_->predict_dist(face, &age_dist);
        DetectionBox box;
        box.x = f.box.x; box.y = f.box.y; box.w = f.box.width; box.h = f.box.height;
        box.score = f.score; box.label_id = f.label_id;
        box.label_name = "face";
        // 性别概率：下标 0=女、1=男（SDK 口径），取男生概率作为 gender_male 分数
        if (!names.empty())
            box.attributes.emplace_back(
                names[0], gender_probs.size() >= 2 ? std::clamp(gender_probs[1], 0.f, 1.f) : 0.f);
        if (names.size() > 1)
            box.attributes.emplace_back(names[1], got_age ? age_young_score(age_dist) : 0.f);
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_face_as(const ImageData& image, InferResult* result) {
    if (!fas_det_model_ || !result) return false;
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    std::vector<KeyPointsResult> faces;
    if (!fas_det_model_->predict(frame, &faces)) return false;
    const float thresh = cfg_.confidence_threshold;
    std::vector<KeyPointsResult> kept;
    kept.reserve(faces.size());
    for (const auto& f : faces)
        if (f.score >= thresh) kept.push_back(f);
    if (kept.empty()) return true;  // 无人脸：无事件对象（不伪造分数）
    // 5 点对齐后送入 fas_first 逐脸/批打分（活体分数 ∈ [0,1]）
    std::vector<ImageData> aligned = utils::align_face_with_five_points(frame, kept);
    std::vector<float> scores;
    if (aligned.size() == 1) {
        scores.assign(1, 0.f);
        fas_first_model_->predict(aligned[0], &scores[0]);
    } else if (!aligned.empty()) {
        fas_first_model_->batch_predict(aligned, &scores);
    }
    // fas_second 可选：任一高置信检测（label>=1 且 score>=0.8）→ 整体判为伪造，liveness=0
    // （对齐 SDK SeetaFaceAsPipeline 的 has_box 语义）
    bool spoof_global = false;
    if (fas_second_model_) {
        std::vector<std::tuple<int, float>> second;
        if (fas_second_model_->predict(frame, &second)) {
            for (const auto& [label, score] : second)
                if (label >= 1 && score >= 0.8f) { spoof_global = true; break; }
        }
    }
    const size_t n = std::min(kept.size(), scores.size());
    for (size_t i = 0; i < n; ++i) {
        DetectionBox box;
        box.x = kept[i].box.x; box.y = kept[i].box.y;
        box.w = kept[i].box.width; box.h = kept[i].box.height;
        box.score = kept[i].score; box.label_id = kept[i].label_id;
        box.label_name = "face";
        // 契约：attributes["liveness"] = 活体分数（低分=非活体）
        box.attributes.emplace_back("liveness",
                                    spoof_global ? 0.f : std::clamp(scores[i], 0.f, 1.f));
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_depth(const ImageData& image, InferResult* result) {
    if (!depth_model_ || !result) return false;
    DepthResult d;
    // UltralyticsDepth::predict 内部支持 NV12/设备帧（零拷贝 NV12 预处路径）
    if (!depth_model_->predict(image, &d)) return false;
    float near = 0.f;
    if (!nearest_depth(d.depth, &near)) return false;  // 全无效深度：不产生对象（fail-closed）
    const float W = static_cast<float>(image.width());
    const float H = static_cast<float>(image.height());
    // 深度模型无实例框：产出整帧对象，depth=帧内最近有效距离（米），供云端 distance 叶子判定
    DetectionBox box;
    box.x = 0.f; box.y = 0.f; box.w = W; box.h = H;
    box.label_id = -1;
    box.score = 0.f;
    box.label_name = "depth";
    box.has_depth = true;
    box.depth = near;
    result->boxes.push_back(std::move(box));
    return true;
}

bool InferenceEngine::infer_sem(const ImageData& image, InferResult* result) {
    if (!sem_model_ || !result) return false;
    SemSegResult sem;
    // UltralyticsSem::predict 内部支持 NV12/设备帧（零拷贝 NV12 预处路径）
    if (!sem_model_->predict(image, &sem)) return false;
    const int64_t total = static_cast<int64_t>(sem.labels.size());
    if (total <= 0) return false;   // 无像素分类结果：不产生对象（fail-closed）
    // 逐像素统计类别占比。num_classes 来自输出通道；labels 若含更大索引则防御性扩展桶。
    int max_idx = 0;
    for (uint8_t v : sem.labels) max_idx = std::max(max_idx, static_cast<int>(v));
    const int ncls = std::max<int>(std::max<int32_t>(sem.num_classes, max_idx + 1), 1);
    std::vector<int64_t> counts(static_cast<size_t>(ncls), 0);
    for (uint8_t v : sem.labels) ++counts[v];
    int dominant = 0;
    for (int c = 1; c < ncls; ++c)
        if (counts[c] > counts[dominant]) dominant = c;
    const float dominant_ratio = static_cast<float>(counts[dominant]) / static_cast<float>(total);
    const float W = static_cast<float>(image.width());
    const float H = static_cast<float>(image.height());
    // 语义分割无实例框：产出整帧对象（归一化 0,0,1,1），attributes={类别名: 面积占比}，
    // 复用分类契约形状（无需云端新增字段）；占比按降序写入，主导类必然在内。
    DetectionBox box;
    box.x = 0.f; box.y = 0.f; box.w = W; box.h = H;
    box.label_id = dominant;
    box.score = dominant_ratio;
    box.label_name = class_name_of(cfg_, dominant);
    std::vector<int> present;
    present.reserve(static_cast<size_t>(ncls));
    for (int c = 0; c < ncls; ++c)
        if (counts[c] > 0) present.push_back(c);
    std::sort(present.begin(), present.end(), [&](int a, int b) { return counts[a] > counts[b]; });
    for (int c : present) {
        const float ratio = static_cast<float>(counts[c]) / static_cast<float>(total);
        box.attributes.emplace_back(class_name_of(cfg_, c), std::clamp(ratio, 0.f, 1.f));
    }
    result->boxes.push_back(std::move(box));
    return true;
}

bool InferenceEngine::infer_face_landmark(const ImageData& image, InferResult* result) {
    if (!face_landmark_model_ || !result) return false;
    // landmark 预处走 image.asMat（packed CPU):NV12（硬解/摄像头帧）先转 BGR，避免越界
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    std::vector<KeyPointsResult> lms;
    if (!face_landmark_model_->predict(frame, &lms)) return false;
    for (const auto& f : lms) {
        if (f.keypoints.empty()) continue;
        // SDK 契约：输入即人脸裁剪图，输出 106 点与该输入同坐标系（事件侧归一化）。
        // 承载框沿用 SDK 返回的 box（=整帧），label 固定 "face"（非空，供节流/过滤）。
        DetectionBox box;
        box.x = f.box.x; box.y = f.box.y;
        box.w = f.box.width; box.h = f.box.height;
        box.score = f.score; box.label_id = f.label_id;
        box.label_name = "face";
        box.keypoints.reserve(f.keypoints.size());
        for (const auto& k : f.keypoints) box.keypoints.push_back({k.x, k.y, k.z});
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_doc(const ImageData& image, InferResult* result) {
    if (!ocr_model_ || !result) return false;
    // 文档/表格：复用 OCR pipeline 抽取文本，逐行写入 objects[].text
    // （云侧 text_match/ocr_label 叶子读取 text；structure 叶子未实现，不伪造结构摘要）
    return infer_ocr(image, result);
}

bool InferenceEngine::infer_barcode(const ImageData& image, InferResult* result) {
    if (!barcode_detector_ || !result) return false;
    // ZXing 仅支持 CPU packed/GRAY；NV12（硬解/摄像头帧）先转 BGR
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    const std::vector<barcode::BarcodeResult> codes = barcode_detector_->detect(frame);
    for (const auto& c : codes) {
        if (c.text.empty()) continue;   // 解码文本为空：不产生对象（不伪造码值）
        // 码区域四角 → 轴对齐外接框（事件契约只有 x/y/w/h，与 OCR/OBB 同口径降维）
        float xmin = c.quad[0].x, ymin = c.quad[0].y;
        float xmax = c.quad[0].x, ymax = c.quad[0].y;
        for (const auto& p : c.quad) {
            xmin = std::min(xmin, p.x); ymin = std::min(ymin, p.y);
            xmax = std::max(xmax, p.x); ymax = std::max(ymax, p.y);
        }
        DetectionBox box;
        box.x = xmin; box.y = ymin;
        box.w = xmax - xmin; box.h = ymax - ymin;
        box.label_id = -1;
        box.score = c.score;
        box.text = c.text;
        box.text_score = c.score;
        // label 用码制名（QR_CODE/EAN_13...）保证非空；供事件节流键与人工排查
        box.label_name = c.format.empty() ? "barcode" : c.format;
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_sam(const ImageData& image, InferResult* result) {
    if (!sam_model_ || !result) return false;
    // FastSAM 预处走 image.asMat（packed CPU）：NV12（硬解/摄像头帧）先转 BGR，避免越界
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    const float W = static_cast<float>(frame.width());
    const float H = static_cast<float>(frame.height());
    std::vector<InstanceSegResult> segs;
    if (!sam_model_->predict(frame, &segs)) return false;
    if (segs.empty()) return true;   // 无实例：不产生对象（fail-closed）
    // 提示点（归一化）→ 像素，映射到「掩码命中该点」的实例（分数最高者）；
    // 无提示点 / 提示点未命中任何掩码 → 回退最高分掩码（兜底，见 select_segment_index）。
    float pt[2] = {0.f, 0.f};
    const bool has_prompt = prompt_to_pixel(cfg_, W, H, pt);
    const int idx = select_segment_index(segs, has_prompt ? pt : nullptr);
    if (idx < 0) return true;
    const auto& s = segs[static_cast<size_t>(idx)];
    // 只送实例框：像素坐标由事件层按帧宽高归一化（与 detection 同契约），云端无需新字段
    DetectionBox box;
    box.x = s.box.x; box.y = s.box.y; box.w = s.box.width; box.h = s.box.height;
    box.score = s.score;
    box.label_id = s.label_id;
    // label 固定 "segment"：云端 prompt_segment/object_present 叶子按此名判定
    box.label_name = "segment";
    result->boxes.push_back(std::move(box));
    return true;
}

bool InferenceEngine::infer_face_rec(const ImageData& image, InferResult* result) {
    if (!fr_det_model_ || !fr_rec_model_ || !result) return false;
    // SCRFD / ArcFace 预处只支持 packed BGR：NV12 帧先转换（同 face_attr/face_as 崩溃修复）
    ImageData frame = to_packed_bgr(image);
    if (frame.empty()) return false;
    std::vector<KeyPointsResult> faces;
    if (!fr_det_model_->predict(frame, &faces)) return false;
    const float thresh = cfg_.confidence_threshold;
    // 逐脸收集 5 点（ArcFace 对齐所需）：缺失/不足 5 点的检测跳过（无法对齐即不产嵌入）
    std::vector<std::vector<std::array<float, 2>>> kps_list;
    std::vector<size_t> kept;
    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        if (f.score < thresh || f.keypoints.size() < 5) continue;
        std::vector<std::array<float, 2>> kps;
        kps.reserve(5);
        for (int k = 0; k < 5; ++k) kps.push_back({f.keypoints[k].x, f.keypoints[k].y});
        kps_list.push_back(std::move(kps));
        kept.push_back(i);
    }
    if (kps_list.empty()) return true;   // 无人脸：无对象（不伪造嵌入）
    std::vector<std::vector<float>> embeddings;
    if (!fr_rec_model_->batch_predict(frame, kps_list, &embeddings)) return false;
    const size_t n = std::min(kept.size(), embeddings.size());
    for (size_t j = 0; j < n; ++j) {
        std::vector<float>& emb = embeddings[j];
        if (!l2_normalize(&emb)) continue;   // 无效嵌入（全零/非有限）丢弃
        const auto& f = faces[kept[j]];
        DetectionBox box;
        box.x = f.box.x; box.y = f.box.y; box.w = f.box.width; box.h = f.box.height;
        box.score = f.score;
        box.label_id = f.label_id;
        box.label_name = "face";
        box.embedding = std::move(emb);   // 已 L2 归一化（512-d，w600k_r50）
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_pedestrian_attribute(const ImageData& image, InferResult* result) {
    if (!pa_model_ || !result) return false;
    std::vector<AttributeResult> attrs;
    if (!pa_model_->predict(image, &attrs)) return false;
    result->attributes = attrs;
    result->boxes.clear();
    for (const auto& a : attrs) {
        DetectionBox b;
        b.x = a.box.x; b.y = a.box.y; b.w = a.box.width; b.h = a.box.height;
        b.score = a.box_score; b.label_id = a.box_label_id;
        const auto& labels = cfg_.labels;
        b.label_name = (a.box_label_id >= 0 && static_cast<size_t>(a.box_label_id) < labels.size())
                           ? labels[a.box_label_id] : "person";
        for (size_t i = 0; i < a.attr_scores.size() && i < cfg_.attributes.size(); ++i)
            b.attributes.emplace_back(cfg_.attributes[i], a.attr_scores[i]);
        result->boxes.push_back(std::move(b));
    }
    return true;
}

bool InferenceEngine::infer_ocr(const ImageData& image, InferResult* result) {
    if (!ocr_model_ || !result) return false;
    OCRResult ocr;
    if (!ocr_model_->predict(image, &ocr)) return false;
    for (size_t i = 0; i < ocr.boxes.size(); ++i) {
        const auto& quad = ocr.boxes[i];  // 4 点：x0,y0,x1,y1,x2,y2,x3,y3
        // 四点最小外接矩形（DetectionBox 为轴对齐坐标，取包围盒）
        int xmin = quad[0], xmax = quad[0];
        int ymin = quad[1], ymax = quad[1];
        for (int k = 1; k < 4; ++k) {
            const int px = quad[k * 2];
            const int py = quad[k * 2 + 1];
            if (px < xmin) xmin = px;
            if (px > xmax) xmax = px;
            if (py < ymin) ymin = py;
            if (py > ymax) ymax = py;
        }
        DetectionBox box;
        // OCR 无类别，显式初始化标签与得分，避免未定义值
        box.label_id = -1;
        box.score = 0.f;
        box.x = static_cast<float>(xmin);
        box.y = static_cast<float>(ymin);
        box.w = static_cast<float>(xmax - xmin);
        box.h = static_cast<float>(ymax - ymin);
        if (i < ocr.rec_scores.size()) {
            box.score = ocr.rec_scores[i];
            box.text_score = ocr.rec_scores[i];
        }
        if (i < ocr.text.size()) box.text = ocr.text[i];
        result->boxes.push_back(std::move(box));
    }
    return true;
}

bool InferenceEngine::infer_lpr(const ImageData& image, InferResult* result) {
    if (!lpr_model_ || !result) return false;
    std::vector<LprResult> lpr_results;
    if (!lpr_model_->predict(image, &lpr_results)) return false;
    for (const auto& p : lpr_results) {
        DetectionBox box;
        box.x = p.box.x; box.y = p.box.y;
        box.w = p.box.width; box.h = p.box.height;
        // 车牌无类别：label_id 固定 -1；label_name 固定 "plate"，保证事件标签/节流键非空
        box.label_id = -1;
        box.label_name = "plate";
        box.score = p.score;
        box.text = p.car_plate_str;
        box.text_score = p.score;
        result->boxes.push_back(std::move(box));
    }
    return true;
}
