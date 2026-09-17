#pragma once
#include <array>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "csrc/vision/detection/ultralytics_det.h"
#include "csrc/vision/classification/classification.h"
#include "csrc/vision/obb/ultralytics_obb.h"
#include "csrc/vision/iseg/ultralytics_seg.h"
#include "csrc/vision/pose/ultralytics_pose.h"
#include "csrc/vision/sem/ultralytics_sem.h"
#include "csrc/vision/landmark/face_landmark.h"
#include "csrc/vision/barcode/barcode.h"
#include "csrc/vision/sam/fastsam.h"
#include "csrc/vision/face/insightface/recognition/insightface_recognition.h"
#include "csrc/vision/face/face_det/scrfd.h"
#include "csrc/vision/face/face_gender/seetaface_gender.h"
#include "csrc/vision/face/face_age/seetaface_age.h"
#include "csrc/vision/face/face_as/face_as_first.h"
#include "csrc/vision/face/face_as/face_as_second.h"
#include "csrc/vision/depth/ultralytics_depth.h"
#include "csrc/vision/pipeline/pedestrian_attribute.h"
#include "csrc/vision/ocr/ppocr.h"
#include "csrc/vision/lpr/lpr_models.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/utils/benchmark.h"

#include "config.hpp"
#include "batched_detector.hpp"

struct DetectionBox {
    float x, y, w, h;
    // 默认初始化，避免某些路径漏填导致读到未定义值
    float score = 0.f;
    int label_id = -1;
    int track_id = -1;                    // 跟踪 ID；-1 = 未分配（默认，不启用跟踪时保持）
    std::string label_name;
    // 属性推理：属性名 -> 得分（pedestrian_attribute 用）
    std::vector<std::pair<std::string, float>> attributes;
    // OCR：识别文字与文字置信度
    std::string text;
    float text_score = 0.f;
    // 关键点（pose）：像素坐标 [x, y, score]（事件侧按帧宽高归一化，见 event_bus.cpp）
    std::vector<std::array<float, 3>> keypoints;
    // 深度（depth 模型）：估算距离（米）；has_depth=true 才写入事件 objects[].depth
    bool has_depth = false;
    float depth = 0.f;
    // 人脸嵌入（face_rec）：已 L2 归一化的 512-d 特征；非空才写入事件 objects[].embedding
    std::vector<float> embedding;
};

/// 人脸关键点（最多 5 个：左右眼/鼻尖/左右嘴角）
struct FaceKeypoint {
    float x, y;
};

struct InferResult {
    std::string model_name;
    std::string type;                 // detection / classification / face_detection / pedestrian_attribute / ocr / lpr / obb / iseg / pose
    std::vector<DetectionBox> boxes;
    // 仅人脸：每个 box 对应一组关键点
    std::vector<std::vector<FaceKeypoint>> keypoints;
    // 仅属性推理：原始属性结果（pedestrian_attribute 用）
    std::vector<modeldeploy::vision::AttributeResult> attributes;
};

/// 推理引擎包装器：每路 Pipeline 独占一个实例
/// 通过 model.clone() 共享底层 ORT Session，但 pre/post processor 各实例独立（线程安全）
class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine() { unload(); }

    /// 从配置加载（会创建新的 ORT Session，适合第一次加载）
    bool load(const ModelConfig& cfg);

    /// 从已有实例克隆 detection 模型（共享 ORT Session），SCRFD独立加载
    /// 适用于多路场景：第一路 load，后续 clone
    bool clone_detection_from(const modeldeploy::vision::detection::UltralyticsDet& proto,
                              const ModelConfig& cfg);

    /// 接管一份已 clone 的 face 模型（共享 Runtime）
    void adopt_face_model(std::unique_ptr<modeldeploy::vision::face::Scrfd> model,
                          const ModelConfig& cfg);

    /// 接管共享批处理检测器（多路共享一个模型实例 + 批推理；GPU 高利用率路径）
    void set_shared_detector(std::shared_ptr<BatchedDetector> det, const ModelConfig& cfg);
    BatchedDetector* shared_detector() const { return shared_det_.get(); }

    void unload();
    bool is_loaded() const { return loaded_; }

    bool infer(const modeldeploy::vision::ImageData& image, InferResult* result);

    /// 获取底层 detection 模型指针（BatchScheduler 需要直接调用 batch_predict）
    modeldeploy::vision::detection::UltralyticsDet* det_model() {
        return shared_det_ ? shared_det_->model() : det_model_.get();
    }

    /// detection 统一推理入口：共享批处理检测器优先（批推理），否则本实例模型
    bool predict_detection(const modeldeploy::vision::ImageData& image,
                           std::vector<modeldeploy::vision::DetectionResult>* out);

    const ModelConfig& config() const { return cfg_; }
    std::pair<int, int> input_size() const {
        return {cfg_.input_size[0], cfg_.input_size[1]};
    }

    static std::string make_cache_key(const ModelConfig& cfg);

private:
    bool loaded_ = false;
    ModelConfig cfg_;

    std::unique_ptr<modeldeploy::vision::detection::UltralyticsDet> det_model_;
    std::unique_ptr<modeldeploy::vision::classification::Classification> cls_model_;
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsObb> obb_model_;
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsSeg> iseg_model_;
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsPose> pose_model_;
    std::unique_ptr<modeldeploy::vision::face::Scrfd> face_model_;
    // face_attr：SCRFD 人脸检测 + 性别/年龄属性（输出概率分数）
    std::unique_ptr<modeldeploy::vision::face::Scrfd> fa_det_model_;
    std::unique_ptr<modeldeploy::vision::face::SeetaFaceGender> fa_gender_model_;
    std::unique_ptr<modeldeploy::vision::face::SeetaFaceAge> fa_age_model_;
    // face_as：SCRFD 人脸检测 + fas_first 活体分数（fas_second 可选）
    std::unique_ptr<modeldeploy::vision::face::Scrfd> fas_det_model_;
    std::unique_ptr<modeldeploy::vision::face::SeetaFaceAsFirst> fas_first_model_;
    std::unique_ptr<modeldeploy::vision::face::SeetaFaceAsSecond> fas_second_model_;
    // depth：单目深度估计（输出整帧最近距离）
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsDepth> depth_model_;
    // sem：语义分割（整帧占比 → attributes{类别名: 占比}）
    std::unique_ptr<modeldeploy::vision::detection::UltralyticsSem> sem_model_;
    // face_landmark：人脸关键点（InsightFace 2d106，106 点；与 pose 共用 keypoints 契约）
    std::unique_ptr<modeldeploy::vision::landmark::FaceLandmark> face_landmark_model_;
    // barcode：条码/二维码原生解码（ZXing，无 DNN 权重）
    std::unique_ptr<modeldeploy::vision::barcode::BarcodeDetector> barcode_detector_;
    // sam：交互式分割（FastSAM-s；prompt_point 命中掩码 → 单个 segment 对象）
    std::unique_ptr<modeldeploy::vision::seg::FastSam> sam_model_;
    // face_rec：SCRFD 人脸检测 + ArcFace 嵌入（w600k_r50，512-d，事件 objects[].embedding）
    std::unique_ptr<modeldeploy::vision::face::Scrfd> fr_det_model_;
    std::unique_ptr<modeldeploy::vision::face::InsightFaceRecognition> fr_rec_model_;
    std::unique_ptr<modeldeploy::vision::pipeline::PedestrianAttribute> pa_model_;
    std::unique_ptr<modeldeploy::vision::ocr::PaddleOCR> ocr_model_;
    std::unique_ptr<modeldeploy::vision::lpr::LprPipeline> lpr_model_;
    std::shared_ptr<BatchedDetector> shared_det_;   // 共享批处理检测器（多路共享，优先）

    bool infer_detection(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_classification(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_obb(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_iseg(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_pose(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face_attr(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face_as(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_depth(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_sem(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face_landmark(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_doc(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_barcode(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_sam(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_face_rec(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_pedestrian_attribute(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_ocr(const modeldeploy::vision::ImageData& image, InferResult* result);
    bool infer_lpr(const modeldeploy::vision::ImageData& image, InferResult* result);
};
