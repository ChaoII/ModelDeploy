//
// ModelDeploy 纯 C API v2 实现
//
#include "md_capi.h"

#include <new>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <array>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_HIGHGUI
#include <opencv2/highgui.hpp>
#endif

#include "csrc/vision.h"
#include "csrc/vision/face/insightface/face_analysis.h"
#include "csrc/vision/face/insightface/insightface_types.h"
#include "csrc/vision/ocr/ppocr.h"
#include "csrc/vision/ocr/dbdetector.h"
#include "csrc/vision/ocr/recognizer.h"
#include "csrc/vision/ocr/classifier.h"
#include "csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h"
#include "csrc/vision/lpr/lpr_det/lpr_det.h"
#include "csrc/vision/lpr/lpr_rec/lpr_rec.h"
#include "csrc/vision/pipeline/pedestrian_attribute.h"
#include "csrc/vision/common/visualize/utils.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/utils/wave_helper.h"
#include "csrc/utils/utils.h"
#include "csrc/core/md_log.h"

#ifdef BUILD_AUDIO
#include "csrc/audio/asr/sense_voice.h"
#include "csrc/audio/tts/kokoro.h"
#endif

/* ---------------- 句柄实现（全局命名空间，与 md_capi.h 前向声明对应） ---------------- */

/* 图像句柄：持有 BGR 数据（owns_data 时库持有）或引用外部内存 */
struct md_image_handle {
    bool owns_data = false;
    int width = 0;
    int height = 0;
    unsigned char* data = nullptr;
    std::unique_ptr<std::vector<unsigned char>> encoded;  // encode 输出暂存
    ~md_image_handle() { if (owns_data) delete[] data; }
};

/* 模型句柄：按 kind 持有具体 C++ 模型；音频结果暂存区也挂这里 */
struct md_model_handle {
    MDModelKind kind;
    bool ready = false;
    std::string name;
    void* model = nullptr;
    modeldeploy::RuntimeOption opt;     // 创建时的配置（clone 复用）
    std::vector<float> audio_buf;       // TTS 输出暂存（零拷贝借用）
    std::string text_buf;               // ASR 输出暂存（零拷贝借用）
    ~md_model_handle();
};

/* 结果句柄：统一容器，data 指向 ResultDataBase（T 由 kind 决定） */
struct md_result_handle {
    MDResultKind kind = MD_RES_DETECTION;
    void* data = nullptr;
    ~md_result_handle();
};

namespace {

/* ---------------- 错误模型（thread_local） ---------------- */
thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : "unknown error"; }
void set_error_fmt(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_last_error = buf;
}

/* ---------------- 泛型结果容器 ---------------- */

struct ResultDataBase {
    virtual ~ResultDataBase() = default;
    virtual size_t count() const = 0;
};

template <typename T>
struct ResultData : ResultDataBase {
    std::vector<T> v;
    size_t count() const override { return v.size(); }
};

/* 单值结果包装（sem_seg / depth / age / gender 等整图/单值结果） */
template <typename T>
struct SingleResult : ResultDataBase {
    T value{};
    size_t count() const override { return 1; }
};

/* 投影容器：把 C++ 结果投影为 blittable 项数组，origin 保留原始容器（所有权转移） */
struct ProjectedResultBase : ResultDataBase {
    virtual ResultDataBase* origin_ptr() const = 0;
};

template <typename Dst>
struct ProjectedResult : ProjectedResultBase {
    std::vector<Dst> v;
    ResultDataBase* origin = nullptr;  // 原始 ResultData<T>*，析构时释放
    size_t count() const override { return v.size(); }
    ResultDataBase* origin_ptr() const override { return origin; }
    ~ProjectedResult() override { delete origin; }
};

// 取结果句柄的原始 C++ 容器（已投影时从 ProjectedResult::origin 取回）
template <typename T>
ResultData<T>* raw_result(md_result_handle* rh) {
    auto* base = static_cast<ResultDataBase*>(rh->data);
    if (auto* d = dynamic_cast<ResultData<T>*>(base)) return d;
    if (auto* p = dynamic_cast<ProjectedResultBase*>(base)) {
        return dynamic_cast<ResultData<T>*>(p->origin_ptr());
    }
    return nullptr;
}

using namespace modeldeploy;
using namespace modeldeploy::vision;

/* 把 ImageHandle 转成 ImageData（零拷贝引用外部 BGR） */
ImageData handle_to_image(const md_image_handle* hi) {
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    return ImageData(mat);
}

/* 按分隔符拆分子模型路径 */
std::vector<std::string> split_path(const std::string& p, char sep = '|') {
    std::vector<std::string> parts;
    size_t s = 0;
    while (true) {
        const size_t sep_pos = p.find(sep, s);
        if (sep_pos == std::string::npos) { parts.push_back(p.substr(s)); break; }
        parts.push_back(p.substr(s, sep_pos - s));
        s = sep_pos + 1;
    }
    return parts;
}

} // namespace

/* ==================== 错误 ==================== */

const char* md_get_last_error(void) {
    return g_last_error.c_str();
}

/* ==================== 选项 ==================== */

struct md_option_handle {
    RuntimeOption opt;
    bool backend_explicit = false;
};

MDStatus md_option_create(MDOptionHandle* out) {
    if (!out) return MD_ERR_NULL_POINTER;
    *out = new md_option_handle();
    return MD_OK;
}

void md_option_destroy(MDOptionHandle h) {
    delete static_cast<md_option_handle*>(h);
}

void md_option_set_device(MDOptionHandle h, MDDevice d) {
    auto* o = static_cast<md_option_handle*>(h);
    switch (d) {
        case MD_DEV_CPU: o->opt.use_cpu(); break;
        case MD_DEV_GPU: o->opt.use_gpu(0); break;
        case MD_DEV_TPU: o->opt.use_sophgo_backend(0); break;
        default: break;
    }
}

void md_option_set_backend(MDOptionHandle h, MDBackend b) {
    auto* o = static_cast<md_option_handle*>(h);
    switch (b) {
        case MD_BK_ORT: o->opt.use_ort_backend(); break;
        case MD_BK_MNN: o->opt.use_mnn_backend(); break;
        case MD_BK_TRT: o->opt.use_trt_backend(); break;
        case MD_BK_SOPHGO: o->opt.use_sophgo_backend(0); break;
    }
    o->backend_explicit = true;
}

void md_option_set_cpu_threads(MDOptionHandle h, int n) {
    static_cast<md_option_handle*>(h)->opt.set_cpu_thread_num(n);
}

void md_option_set_fp16(MDOptionHandle h, int enable) {
    static_cast<md_option_handle*>(h)->opt.enable_fp16 = enable != 0;
}

void md_option_set_trt_engine_path(MDOptionHandle h, const char* path) {
    static_cast<md_option_handle*>(h)->opt.ort_option.trt_engine_cache_path = path ? path : "";
}

/* ==================== 图像 ==================== */

MDStatus md_image_from_file(MDImageHandle* out, const char* path) {
    if (!out) return MD_ERR_NULL_POINTER;
    if (!path || !*path) { set_error("md_image_from_file: path is empty"); return MD_ERR_INVALID_ARGUMENT; }
    auto img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) { set_error_fmt("md_image_from_file: cannot decode '%s'", path); return MD_ERR_IMAGE_DECODE; }
    auto* h = new md_image_handle();
    h->width = img.cols;
    h->height = img.rows;
    const size_t bytes = static_cast<size_t>(img.total()) * 3;
    h->data = new unsigned char[bytes];
    std::memcpy(h->data, img.data, bytes);
    h->owns_data = true;
    *out = h;
    return MD_OK;
}

static MDStatus image_from_mat(MDImageHandle* out, cv::Mat&& mat) {
    if (!out) return MD_ERR_NULL_POINTER;
    if (mat.empty()) return MD_ERR_IMAGE_DECODE;
    auto* h = new md_image_handle();
    h->width = mat.cols;
    h->height = mat.rows;
    const size_t bytes = static_cast<size_t>(mat.total()) * 3;
    h->data = new unsigned char[bytes];
    std::memcpy(h->data, mat.data, bytes);
    h->owns_data = true;
    *out = h;
    return MD_OK;
}

MDStatus md_image_from_bgr24(MDImageHandle* out, const void* bgr, int w, int h) {
    if (!out || !bgr) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_bgr24: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    auto* hi = new md_image_handle();
    hi->width = w;
    hi->height = h;
    hi->data = static_cast<unsigned char*>(const_cast<void*>(bgr));
    *out = hi;
    return MD_OK;
}

MDStatus md_image_from_rgb24(MDImageHandle* out, const void* rgb, int w, int h) {
    if (!out || !rgb) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_rgb24: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    cv::Mat src(h, w, CV_8UC3, const_cast<void*>(rgb));
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
    return image_from_mat(out, std::move(bgr));
}

MDStatus md_image_from_nv12(MDImageHandle* out, const void* y, const void* uv,
                            int w, int h, int step_y, int step_uv, MDDevice src) {
    if (!out || !y) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_nv12: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    if (step_y <= 0) step_y = w;
    if (step_uv <= 0) step_uv = w;
    (void)src;
    const int uv_h = h / 2;
    cv::Mat y_mat(h, step_y, CV_8UC1, const_cast<void*>(y));
    cv::Mat uv_mat(uv_h, step_uv, CV_8UC2, const_cast<void*>(uv));
    cv::Mat bgr;
    cv::cvtColorTwoPlane(y_mat(cv::Rect(0, 0, w, h)), uv_mat(cv::Rect(0, 0, w, uv_h)),
                         bgr, cv::COLOR_YUV2BGR_NV12);
    return image_from_mat(out, std::move(bgr));
}

MDStatus md_image_from_yuv420p(MDImageHandle* out, const void* data, int w, int h) {
    if (!out || !data) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_yuv420p: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    cv::Mat yuv(h * 3 / 2, w, CV_8UC1, const_cast<void*>(data));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
    return image_from_mat(out, std::move(bgr));
}

MDStatus md_image_from_encoded(MDImageHandle* out, const void* bytes, size_t n) {
    if (!out || !bytes) return MD_ERR_NULL_POINTER;
    if (n == 0) return MD_ERR_INVALID_ARGUMENT;
    const auto* p = static_cast<const unsigned char*>(bytes);
    const std::vector<unsigned char> buf(p, p + n);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) return MD_ERR_IMAGE_DECODE;
    return image_from_mat(out, std::move(img));
}

MDStatus md_image_from_base64(MDImageHandle* out, const char* b64) {
    if (!out || !b64) return MD_ERR_NULL_POINTER;
    const std::vector<unsigned char> buf = modeldeploy::base64_decode(b64);
    if (buf.empty()) return MD_ERR_IMAGE_DECODE;
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) return MD_ERR_IMAGE_DECODE;
    return image_from_mat(out, std::move(img));
}

MDStatus md_image_clone(MDImageHandle in, MDImageHandle* out) {
    if (!in || !out) return MD_ERR_NULL_POINTER;
    const auto* hi = static_cast<md_image_handle*>(in);
    auto* nh = new md_image_handle();
    nh->width = hi->width;
    nh->height = hi->height;
    const size_t bytes = static_cast<size_t>(hi->width) * hi->height * 3;
    nh->data = new unsigned char[bytes];
    std::memcpy(nh->data, hi->data, bytes);
    nh->owns_data = true;
    *out = nh;
    return MD_OK;
}

MDStatus md_image_crop(MDImageHandle in, int x, int y, int w, int h, MDImageHandle* out) {
    if (!in || !out) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) return MD_ERR_INVALID_ARGUMENT;
    const auto* hi = static_cast<md_image_handle*>(in);
    cv::Mat src(hi->height, hi->width, CV_8UC3, hi->data);
    if (x < 0 || y < 0 || x + w > hi->width || y + h > hi->height) {
        set_error("md_image_crop: crop rect out of bounds");
        return MD_ERR_INVALID_ARGUMENT;
    }
    return image_from_mat(out, src(cv::Rect(x, y, w, h)).clone());
}

MDStatus md_image_show(MDImageHandle h) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi) return MD_ERR_NULL_POINTER;
#ifdef HAVE_OPENCV_HIGHGUI
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    cv::imshow("ModelDeploy", mat);
    cv::waitKey(0);
    return MD_OK;
#else
    set_error("md_image_show: OpenCV built without highgui");
    return MD_ERR_NOT_IMPLEMENTED;
#endif
}

MDStatus md_image_save(MDImageHandle h, const char* path) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi || !path || !*path) return MD_ERR_NULL_POINTER;
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    if (!cv::imwrite(path, mat)) { set_error_fmt("md_image_save: failed to write '%s'", path); return MD_ERR_INVALID_ARGUMENT; }
    return MD_OK;
}

MDStatus md_image_encode(MDImageHandle h, const char* ext,
                         const unsigned char** buf, size_t* n) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi || !ext || !buf || !n) return MD_ERR_NULL_POINTER;
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    std::vector<int> params;
    if (std::strcmp(ext, ".jpg") == 0 || std::strcmp(ext, ".jpeg") == 0) params = {cv::IMWRITE_JPEG_QUALITY, 95};
    else if (std::strcmp(ext, ".png") == 0) params = {cv::IMWRITE_PNG_COMPRESSION, 3};
    auto encoded = std::make_unique<std::vector<unsigned char>>();
    if (!cv::imencode(ext, mat, *encoded, params)) return MD_ERR_INVALID_ARGUMENT;
    *buf = encoded->data();
    *n = encoded->size();
    hi->encoded = std::move(encoded);
    return MD_OK;
}

void md_image_destroy(MDImageHandle h) {
    delete static_cast<md_image_handle*>(h);
}

MDStatus md_image_size(MDImageHandle h, int* w, int* height_out) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi) return MD_ERR_NULL_POINTER;
    if (w) *w = hi->width;
    if (height_out) *height_out = hi->height;
    return MD_OK;
}

/* ==================== 模型创建分发 ==================== */

namespace {

template <typename M>
M* make_model(const char* model_path, const RuntimeOption& opt,
              const std::string& what, std::string* err) {
    auto* m = new M(model_path, opt);
    if (!m->is_initialized()) {
        *err = what + " failed to initialize";
        delete m;
        return nullptr;
    }
    return m;
}

} // namespace

MDStatus md_model_create(MDModelHandle* out, MDModelKind kind,
                         const char* model_path, const MDOptionHandle opt_h) {
    if (!out) return MD_ERR_NULL_POINTER;
    if (!model_path || !*model_path) { set_error("md_model_create: model path is empty"); return MD_ERR_INVALID_ARGUMENT; }

    auto* mh = new md_model_handle();
    mh->kind = kind;
    mh->name = model_path;

    const RuntimeOption& opt = opt_h ? static_cast<const md_option_handle*>(opt_h)->opt : RuntimeOption();
    mh->opt = opt;
    std::string err;
    auto fail_init = [&](const char* what) {
        set_error_fmt("md_model_create: %s failed to initialize: %s", what, model_path);
        delete mh;
        return MD_ERR_MODEL_INIT;
    };
    auto need_parts = [&](size_t n, const char* what) -> bool {
        if (split_path(model_path).size() >= n) return true;
        set_error_fmt("md_model_create: %s needs %zu parts separated by '|'", what, n);
        delete mh;
        return false;
    };

    switch (kind) {
        case MD_MODEL_DETECTION: {
            mh->model = make_model<detection::UltralyticsDet>(model_path, opt, "UltralyticsDet", &err);
            if (!mh->model) return fail_init("UltralyticsDet");
            break;
        }
        case MD_MODEL_CLASSIFICATION: {
            mh->model = make_model<classification::Classification>(model_path, opt, "Classification", &err);
            if (!mh->model) return fail_init("Classification");
            break;
        }
        case MD_MODEL_POSE: {
            mh->model = make_model<detection::UltralyticsPose>(model_path, opt, "UltralyticsPose", &err);
            if (!mh->model) return fail_init("UltralyticsPose");
            break;
        }
        case MD_MODEL_OBB: {
            mh->model = make_model<detection::UltralyticsObb>(model_path, opt, "UltralyticsObb", &err);
            if (!mh->model) return fail_init("UltralyticsObb");
            break;
        }
        case MD_MODEL_INSTANCE_SEG: {
            mh->model = make_model<detection::UltralyticsSeg>(model_path, opt, "UltralyticsSeg", &err);
            if (!mh->model) return fail_init("UltralyticsSeg");
            break;
        }
        case MD_MODEL_SEM_SEG: {
            mh->model = make_model<detection::UltralyticsSem>(model_path, opt, "UltralyticsSem", &err);
            if (!mh->model) return fail_init("UltralyticsSem");
            break;
        }
        case MD_MODEL_DEPTH: {
            mh->model = make_model<detection::UltralyticsDepth>(model_path, opt, "UltralyticsDepth", &err);
            if (!mh->model) return fail_init("UltralyticsDepth");
            break;
        }
        case MD_MODEL_FACE_DET: {
            mh->model = make_model<face::Scrfd>(model_path, opt, "Scrfd", &err);
            if (!mh->model) return fail_init("Scrfd");
            break;
        }
        case MD_MODEL_FACE_REC: {
            mh->model = make_model<face::SeetaFaceID>(model_path, opt, "SeetaFaceID", &err);
            if (!mh->model) return fail_init("SeetaFaceID");
            break;
        }
        case MD_MODEL_FACE_AGE: {
            mh->model = make_model<face::SeetaFaceAge>(model_path, opt, "SeetaFaceAge", &err);
            if (!mh->model) return fail_init("SeetaFaceAge");
            break;
        }
        case MD_MODEL_FACE_GENDER: {
            mh->model = make_model<face::SeetaFaceGender>(model_path, opt, "SeetaFaceGender", &err);
            if (!mh->model) return fail_init("SeetaFaceGender");
            break;
        }
        case MD_MODEL_FACE_AS: {
            mh->model = make_model<face::InsightFaceGenderAge>(model_path, opt, "InsightFaceGenderAge", &err);
            if (!mh->model) return fail_init("InsightFaceGenderAge");
            break;
        }
        case MD_MODEL_INSIGHTFACE_DET: {
            mh->model = make_model<face::InsightFaceDet>(model_path, opt, "InsightFaceDet", &err);
            if (!mh->model) return fail_init("InsightFaceDet");
            break;
        }
        case MD_MODEL_FACE_REC_PIPELINE: {
            if (!need_parts(2, "face-rec-pipeline")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new face::FaceRecognizerPipeline(parts[0], parts[1], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("FaceRecognizerPipeline");
            break;
        }
        case MD_MODEL_INSIGHTFACE: {
            if (!need_parts(4, "insightface")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new face::InsightFaceAnalysis(parts[0], parts[1], parts[2], parts[3], opt,
                                                    parts.size() > 4 ? parts[4] : "");
            mh->model = m;
            if (!m->is_initialized()) return fail_init("InsightFaceAnalysis");
            break;
        }
        case MD_MODEL_OCR: {
            if (!need_parts(4, "ocr")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new ocr::PaddleOCR(parts[0], parts[1], parts[2], parts[3], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("PaddleOCR");
            break;
        }
        case MD_MODEL_OCR_DET: {
            mh->model = make_model<ocr::DBDetector>(model_path, opt, "DBDetector", &err);
            if (!mh->model) return fail_init("DBDetector");
            break;
        }
        case MD_MODEL_OCR_REC: {
            if (!need_parts(2, "ocr-rec")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new ocr::Recognizer(parts[0], parts[1], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("Recognizer");
            break;
        }
        case MD_MODEL_OCR_CLS: {
            mh->model = make_model<ocr::Classifier>(model_path, opt, "Classifier", &err);
            if (!mh->model) return fail_init("Classifier");
            break;
        }
        case MD_MODEL_LPR_DET: {
            mh->model = make_model<lpr::LprDetection>(model_path, opt, "LprDetection", &err);
            if (!mh->model) return fail_init("LprDetection");
            break;
        }
        case MD_MODEL_LPR_REC: {
            mh->model = make_model<lpr::LprRecognizer>(model_path, opt, "LprRecognizer", &err);
            if (!mh->model) return fail_init("LprRecognizer");
            break;
        }
        case MD_MODEL_LPR_PIPELINE: {
            if (!need_parts(2, "lpr-pipeline")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new lpr::LprPipeline(parts[0], parts[1], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("LprPipeline");
            break;
        }
        case MD_MODEL_PED_ATTR: {
            if (!need_parts(2, "ped-attr")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new pipeline::PedestrianAttribute(parts[0], parts[1], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("PedestrianAttribute");
            break;
        }
#ifdef BUILD_AUDIO
        case MD_MODEL_ASR: {
            if (!need_parts(2, "asr")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new audio::asr::SenseVoice(parts[0], parts[1], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("SenseVoice");
            break;
        }
        case MD_MODEL_TTS: {
            if (!need_parts(7, "tts")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            std::vector<std::string> lexicons;
            lexicons.push_back(parts[2]);
            lexicons.push_back(parts[3]);
            auto* m = new audio::tts::Kokoro(parts[0], parts[1], lexicons,
                                             parts[4], parts[5], parts[6], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("Kokoro");
            break;
        }
#else
        case MD_MODEL_ASR:
        case MD_MODEL_TTS:
            set_error_fmt("md_model_create: kind %d (audio) requires BUILD_AUDIO", (int)kind);
            delete mh;
            return MD_ERR_UNSUPPORTED_TYPE;
#endif
        default:
            set_error_fmt("md_model_create: unsupported kind %d", (int)kind);
            delete mh;
            return MD_ERR_UNSUPPORTED_TYPE;
    }

    mh->ready = true;
    *out = mh;
    return MD_OK;
}

/* ==================== 模型释放 ==================== */

md_model_handle::~md_model_handle() {
    if (!model) return;
    switch (kind) {
        case MD_MODEL_DETECTION: delete static_cast<detection::UltralyticsDet*>(model); break;
        case MD_MODEL_CLASSIFICATION: delete static_cast<classification::Classification*>(model); break;
        case MD_MODEL_POSE: delete static_cast<detection::UltralyticsPose*>(model); break;
        case MD_MODEL_OBB: delete static_cast<detection::UltralyticsObb*>(model); break;
        case MD_MODEL_INSTANCE_SEG: delete static_cast<detection::UltralyticsSeg*>(model); break;
        case MD_MODEL_SEM_SEG: delete static_cast<detection::UltralyticsSem*>(model); break;
        case MD_MODEL_DEPTH: delete static_cast<detection::UltralyticsDepth*>(model); break;
        case MD_MODEL_FACE_DET: delete static_cast<face::Scrfd*>(model); break;
        case MD_MODEL_FACE_REC: delete static_cast<face::SeetaFaceID*>(model); break;
        case MD_MODEL_FACE_AGE: delete static_cast<face::SeetaFaceAge*>(model); break;
        case MD_MODEL_FACE_GENDER: delete static_cast<face::SeetaFaceGender*>(model); break;
        case MD_MODEL_FACE_AS: delete static_cast<face::InsightFaceGenderAge*>(model); break;
        case MD_MODEL_FACE_REC_PIPELINE: delete static_cast<face::FaceRecognizerPipeline*>(model); break;
        case MD_MODEL_INSIGHTFACE: delete static_cast<face::InsightFaceAnalysis*>(model); break;
        case MD_MODEL_INSIGHTFACE_DET: delete static_cast<face::InsightFaceDet*>(model); break;
        case MD_MODEL_OCR: delete static_cast<ocr::PaddleOCR*>(model); break;
        case MD_MODEL_OCR_DET: delete static_cast<ocr::DBDetector*>(model); break;
        case MD_MODEL_OCR_REC: delete static_cast<ocr::Recognizer*>(model); break;
        case MD_MODEL_OCR_CLS: delete static_cast<ocr::Classifier*>(model); break;
        case MD_MODEL_LPR_DET: delete static_cast<lpr::LprDetection*>(model); break;
        case MD_MODEL_LPR_REC: delete static_cast<lpr::LprRecognizer*>(model); break;
        case MD_MODEL_LPR_PIPELINE: delete static_cast<lpr::LprPipeline*>(model); break;
        case MD_MODEL_PED_ATTR: delete static_cast<pipeline::PedestrianAttribute*>(model); break;
#ifdef BUILD_AUDIO
        case MD_MODEL_ASR: delete static_cast<audio::asr::SenseVoice*>(model); break;
        case MD_MODEL_TTS: delete static_cast<audio::tts::Kokoro*>(model); break;
#endif
        default: break;
    }
}

void md_model_destroy(MDModelHandle h) {
    delete static_cast<md_model_handle*>(h);
}

MDStatus md_model_ready(MDModelHandle h) {
    auto* mh = static_cast<md_model_handle*>(h);
    return (mh && mh->ready) ? MD_OK : MD_ERR_MODEL_INIT;
}

MDStatus md_model_clone(MDModelHandle in, MDModelHandle* out) {
    auto* src = static_cast<md_model_handle*>(in);
    if (!src || !out) return MD_ERR_NULL_POINTER;
    if (!src->ready) return MD_ERR_MODEL_INIT;

    // 先尝试用模型内部 clone() 复用已加载的运行时（更快）
    auto* nh = new md_model_handle();
    nh->kind = src->kind;
    nh->name = src->name;
    nh->opt = src->opt;
    void* cloned = nullptr;

    switch (src->kind) {
        case MD_MODEL_DETECTION: cloned = static_cast<detection::UltralyticsDet*>(src->model)->clone().release(); break;
        case MD_MODEL_CLASSIFICATION: cloned = static_cast<classification::Classification*>(src->model)->clone().release(); break;
        case MD_MODEL_POSE: cloned = static_cast<detection::UltralyticsPose*>(src->model)->clone().release(); break;
        case MD_MODEL_OBB: cloned = static_cast<detection::UltralyticsObb*>(src->model)->clone().release(); break;
        case MD_MODEL_INSTANCE_SEG: cloned = static_cast<detection::UltralyticsSeg*>(src->model)->clone().release(); break;
        case MD_MODEL_SEM_SEG: cloned = static_cast<detection::UltralyticsSem*>(src->model)->clone().release(); break;
        case MD_MODEL_DEPTH: cloned = static_cast<detection::UltralyticsDepth*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_DET: cloned = static_cast<face::Scrfd*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_REC: cloned = static_cast<face::SeetaFaceID*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AGE: cloned = static_cast<face::SeetaFaceAge*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_GENDER: cloned = static_cast<face::SeetaFaceGender*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_REC_PIPELINE: cloned = static_cast<face::FaceRecognizerPipeline*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AS: cloned = static_cast<face::InsightFaceGenderAge*>(src->model)->clone().release(); break;
        case MD_MODEL_INSIGHTFACE_DET: cloned = static_cast<face::InsightFaceDet*>(src->model)->clone().release(); break;
        case MD_MODEL_INSIGHTFACE: cloned = static_cast<face::InsightFaceAnalysis*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR: cloned = static_cast<ocr::PaddleOCR*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_DET: cloned = static_cast<ocr::DBDetector*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_REC: cloned = static_cast<ocr::Recognizer*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_CLS: cloned = static_cast<ocr::Classifier*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_DET: cloned = static_cast<lpr::LprDetection*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_REC: cloned = static_cast<lpr::LprRecognizer*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_PIPELINE: cloned = static_cast<lpr::LprPipeline*>(src->model)->clone().release(); break;
        case MD_MODEL_PED_ATTR: cloned = static_cast<pipeline::PedestrianAttribute*>(src->model)->clone().release(); break;
#ifdef BUILD_AUDIO
        case MD_MODEL_ASR: cloned = static_cast<audio::asr::SenseVoice*>(src->model)->clone().release(); break;
        case MD_MODEL_TTS: cloned = static_cast<audio::tts::Kokoro*>(src->model)->clone().release(); break;
#endif
        default: break;
    }

    if (!cloned) {
        // 兜底：某模型未实现 clone() 时用保存的 name+opt 重新加载
        delete nh;
        md_option_handle tmp_opt{};
        tmp_opt.opt = src->opt;
        return md_model_create(out, src->kind, src->name.c_str(), &tmp_opt);
    }

    nh->model = cloned;
    nh->ready = true;
    *out = nh;
    return MD_OK;
}

MDStatus md_model_set_input_size(MDModelHandle handle, int w, int h) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (w <= 0 || h <= 0) return MD_ERR_INVALID_ARGUMENT;
    const std::vector<int> size{w, h};
    switch (mh->kind) {
        case MD_MODEL_DETECTION: static_cast<detection::UltralyticsDet*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_CLASSIFICATION: static_cast<classification::Classification*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_POSE: static_cast<detection::UltralyticsPose*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_OBB: static_cast<detection::UltralyticsObb*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_INSTANCE_SEG: static_cast<detection::UltralyticsSeg*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_SEM_SEG: static_cast<detection::UltralyticsSem*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_DEPTH: static_cast<detection::UltralyticsDepth*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_FACE_DET: static_cast<face::Scrfd*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_PED_ATTR: static_cast<pipeline::PedestrianAttribute*>(mh->model)->set_det_input_size(size); break;
        default:
            set_error_fmt("md_model_set_input_size: unsupported for kind %d", (int)mh->kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

MDStatus md_model_set_cls_input_size(MDModelHandle handle, int w, int h) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (w <= 0 || h <= 0) return MD_ERR_INVALID_ARGUMENT;
    const std::vector<int> size{w, h};
    switch (mh->kind) {
        case MD_MODEL_PED_ATTR: static_cast<pipeline::PedestrianAttribute*>(mh->model)->set_cls_input_size(size); break;
        default:
            set_error_fmt("md_model_set_cls_input_size: unsupported for kind %d", (int)mh->kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

/* ==================== 结果容器释放 ==================== */

md_result_handle::~md_result_handle() {
    delete static_cast<ResultDataBase*>(data);
}

void md_result_destroy(MDResultHandle h) {
    delete static_cast<md_result_handle*>(h);
}

MDStatus md_result_kind(MDResultHandle h, MDResultKind* out) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !out) return MD_ERR_NULL_POINTER;
    *out = rh->kind;
    return MD_OK;
}

MDStatus md_result_count(MDResultHandle h, size_t* out) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !out) return MD_ERR_NULL_POINTER;
    *out = static_cast<ResultDataBase*>(rh->data)->count();
    return MD_OK;
}

/* ==================== 推理分发（视觉） ==================== */

MDStatus md_model_predict(MDModelHandle h, MDImageHandle img_h, MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !img_h || !out) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    const ImageData image = handle_to_image(static_cast<md_image_handle*>(img_h));

    auto* rh = new md_result_handle();

    auto predict_fail = [&](const char* what) {
        set_error_fmt("md_model_predict: %s failed", what);
        delete rh;
        return MD_ERR_MODEL_PREDICT;
    };

    switch (mh->kind) {
        case MD_MODEL_DETECTION: {
            auto* m = static_cast<detection::UltralyticsDet*>(mh->model);
            auto* d = new ResultData<DetectionResult>();
            if (!m->predict(image, &d->v)) return predict_fail("detection");
            rh->kind = MD_RES_DETECTION;
            rh->data = d;
            break;
        }
        case MD_MODEL_CLASSIFICATION: {
            auto* m = static_cast<classification::Classification*>(mh->model);
            auto* d = new ResultData<ClassifyResult>();
            ClassifyResult r;
            if (!m->predict(image, &r)) return predict_fail("classification");
            d->v.push_back(std::move(r));
            rh->kind = MD_RES_CLASSIFICATION;
            rh->data = d;
            break;
        }
        case MD_MODEL_POSE: {
            auto* m = static_cast<detection::UltralyticsPose*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("pose");
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_OBB: {
            auto* m = static_cast<detection::UltralyticsObb*>(mh->model);
            auto* d = new ResultData<ObbResult>();
            if (!m->predict(image, &d->v)) return predict_fail("obb");
            rh->kind = MD_RES_OBB;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSTANCE_SEG: {
            auto* m = static_cast<detection::UltralyticsSeg*>(mh->model);
            auto* d = new ResultData<InstanceSegResult>();
            if (!m->predict(image, &d->v)) return predict_fail("instance seg");
            rh->kind = MD_RES_INSTANCE_SEG;
            rh->data = d;
            break;
        }
        case MD_MODEL_SEM_SEG: {
            auto* m = static_cast<detection::UltralyticsSem*>(mh->model);
            auto* d = new SingleResult<SemSegResult>();
            if (!m->predict(image, &d->value)) return predict_fail("sem seg");
            rh->kind = MD_RES_SEM_SEG;
            rh->data = d;
            break;
        }
        case MD_MODEL_DEPTH: {
            auto* m = static_cast<detection::UltralyticsDepth*>(mh->model);
            auto* d = new SingleResult<DepthResult>();
            if (!m->predict(image, &d->value)) return predict_fail("depth");
            rh->kind = MD_RES_DEPTH;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_DET: {
            auto* m = static_cast<face::Scrfd*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("face det");
            rh->kind = MD_RES_FACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_REC: {
            auto* m = static_cast<face::SeetaFaceID*>(mh->model);
            auto* d = new ResultData<FaceRecognitionResult>();
            FaceRecognitionResult r;
            if (!m->predict(image, &r)) return predict_fail("face rec");
            d->v.push_back(std::move(r));
            rh->kind = MD_RES_FACE_REC;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AGE: {
            auto* m = static_cast<face::SeetaFaceAge*>(mh->model);
            auto* d = new SingleResult<int>();
            int age = 0;
            if (!m->predict(image, &age)) return predict_fail("face age");
            d->value = age;
            rh->kind = MD_RES_AGE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_GENDER: {
            auto* m = static_cast<face::SeetaFaceGender*>(mh->model);
            auto* d = new SingleResult<int>();
            int gender = 0;
            if (!m->predict(image, &gender)) return predict_fail("face gender");
            d->value = gender;
            rh->kind = MD_RES_GENDER;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AS: {
            auto* m = static_cast<face::InsightFaceGenderAge*>(mh->model);
            auto* d = new ResultData<face::InsightFaceResult>();
            face::GenderAgeResult r;
            std::array<float, 4> whole{0.f, 0.f, static_cast<float>(image.width()), static_cast<float>(image.height())};
            if (!m->predict_gender_age(image, whole, &r)) return predict_fail("face gender age");
            face::InsightFaceResult out;
            out.gender = r.gender;
            out.age = r.age;
            out.bbox = whole;
            d->v.push_back(std::move(out));
            rh->kind = MD_RES_INSIGHTFACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_REC_PIPELINE: {
            auto* m = static_cast<face::FaceRecognizerPipeline*>(mh->model);
            auto* d = new ResultData<FaceRecognitionResult>();
            if (!m->predict(image, &d->v)) return predict_fail("face rec pipeline");
            rh->kind = MD_RES_FACE_REC;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSIGHTFACE: {
            auto* m = static_cast<face::InsightFaceAnalysis*>(mh->model);
            auto* d = new ResultData<face::InsightFaceResult>();
            if (!m->analyze(image, &d->v)) return predict_fail("insightface");
            rh->kind = MD_RES_INSIGHTFACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSIGHTFACE_DET: {
            auto* m = static_cast<face::InsightFaceDet*>(mh->model);
            auto* d = new ResultData<face::InsightFaceBox>();
            if (!m->predict(image, &d->v)) return predict_fail("insightface det");
            rh->kind = MD_RES_FACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_OCR: {
            auto* m = static_cast<ocr::PaddleOCR*>(mh->model);
            auto* d = new SingleResult<OCRResult>();
            if (!m->predict(image, &d->value)) return predict_fail("ocr");
            rh->kind = MD_RES_OCR;
            rh->data = d;
            break;
        }
        case MD_MODEL_OCR_DET: {
            auto* m = static_cast<ocr::DBDetector*>(mh->model);
            auto* d = new SingleResult<OCRResult>();
            if (!m->predict(image, &d->value)) return predict_fail("ocr det");
            rh->kind = MD_RES_OCR;
            rh->data = d;
            break;
        }
        case MD_MODEL_OCR_REC: {
            auto* m = static_cast<ocr::Recognizer*>(mh->model);
            auto* d = new SingleResult<OCRResult>();
            if (!m->predict(image, &d->value)) return predict_fail("ocr rec");
            rh->kind = MD_RES_OCR;
            rh->data = d;
            break;
        }
        case MD_MODEL_OCR_CLS: {
            auto* m = static_cast<ocr::Classifier*>(mh->model);
            auto* d = new SingleResult<OCRResult>();
            if (!m->predict(image, &d->value)) return predict_fail("ocr cls");
            rh->kind = MD_RES_OCR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_DET: {
            auto* m = static_cast<lpr::LprDetection*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("lpr det");
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_REC: {
            auto* m = static_cast<lpr::LprRecognizer*>(mh->model);
            auto* d = new ResultData<LprResult>();
            LprResult r;
            if (!m->predict(image, &r)) return predict_fail("lpr rec");
            d->v.push_back(std::move(r));
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_PIPELINE: {
            auto* m = static_cast<lpr::LprPipeline*>(mh->model);
            auto* d = new ResultData<LprResult>();
            if (!m->predict(image, &d->v)) return predict_fail("lpr pipeline");
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_PED_ATTR: {
            auto* m = static_cast<pipeline::PedestrianAttribute*>(mh->model);
            auto* d = new ResultData<AttributeResult>();
            if (!m->predict(image, &d->v)) return predict_fail("ped attr");
            rh->kind = MD_RES_ATTR;
            rh->data = d;
            break;
        }
        default:
            set_error_fmt("md_model_predict: predict not implemented for kind %d", (int)mh->kind);
            delete rh;
            return MD_ERR_NOT_IMPLEMENTED;
    }

    *out = rh;
    return MD_OK;
}

MDStatus md_model_predict_nv12(MDModelHandle handle,
                               const void* y, const void* uv,
                               int w, int h, int step_y, int step_uv,
                               MDDevice src_device, MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !y || !uv || !out) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    if (w <= 0 || h <= 0) return MD_ERR_INVALID_ARGUMENT;
    if (step_y <= 0) step_y = w;
    if (step_uv <= 0) step_uv = w;

    const auto* py = static_cast<const uint8_t*>(y);
    const auto* puv = static_cast<const uint8_t*>(uv);
    const Device dev = [&]() {
        switch (src_device) {
            case MD_DEV_GPU: return Device::GPU;
            case MD_DEV_TPU: return Device::TPU;
            case MD_DEV_CPU:
            default: return Device::CPU;
        }
    }();

    auto* rh = new md_result_handle();

    auto fail = [&](const char* what) {
        set_error_fmt("md_model_predict_nv12: %s failed", what);
        delete rh;
        return MD_ERR_MODEL_PREDICT;
    };

    try {
        switch (mh->kind) {
            case MD_MODEL_DETECTION: {
                auto* m = static_cast<detection::UltralyticsDet*>(mh->model);
                auto* d = new ResultData<DetectionResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->v, nullptr, nullptr, dev))
                    return fail("detection nv12");
                rh->kind = MD_RES_DETECTION;
                rh->data = d;
                break;
            }
            case MD_MODEL_POSE: {
                auto* m = static_cast<detection::UltralyticsPose*>(mh->model);
                auto* d = new ResultData<KeyPointsResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->v, nullptr, nullptr, dev))
                    return fail("pose nv12");
                rh->kind = MD_RES_POSE;
                rh->data = d;
                break;
            }
            case MD_MODEL_OBB: {
                auto* m = static_cast<detection::UltralyticsObb*>(mh->model);
                auto* d = new ResultData<ObbResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->v, nullptr, nullptr, dev))
                    return fail("obb nv12");
                rh->kind = MD_RES_OBB;
                rh->data = d;
                break;
            }
            case MD_MODEL_INSTANCE_SEG: {
                auto* m = static_cast<detection::UltralyticsSeg*>(mh->model);
                auto* d = new ResultData<InstanceSegResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->v, nullptr, nullptr, dev))
                    return fail("instance seg nv12");
                rh->kind = MD_RES_INSTANCE_SEG;
                rh->data = d;
                break;
            }
            case MD_MODEL_SEM_SEG: {
                auto* m = static_cast<detection::UltralyticsSem*>(mh->model);
                auto* d = new SingleResult<SemSegResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->value, nullptr, nullptr, dev))
                    return fail("sem seg nv12");
                rh->kind = MD_RES_SEM_SEG;
                rh->data = d;
                break;
            }
            case MD_MODEL_DEPTH: {
                auto* m = static_cast<detection::UltralyticsDepth*>(mh->model);
                auto* d = new SingleResult<DepthResult>();
                if (!m->predict_nv12(py, puv, w, h, step_y, step_uv, &d->value, nullptr, nullptr, dev))
                    return fail("depth nv12");
                rh->kind = MD_RES_DEPTH;
                rh->data = d;
                break;
            }
            default:
                set_error_fmt("md_model_predict_nv12: unsupported kind %d", (int)mh->kind);
                delete rh;
                return MD_ERR_UNSUPPORTED_TYPE;
        }
    } catch (const std::exception& e) {
        set_error_fmt("md_model_predict_nv12: %s", e.what());
        delete rh;
        return MD_ERR_MODEL_PREDICT;
    }

    *out = rh;
    return MD_OK;
}

MDStatus md_model_predict_batch(MDModelHandle, MDImageHandle*, size_t, MDResultHandle*) {
    set_error("md_model_predict_batch: not implemented in v2");
    return MD_ERR_NOT_IMPLEMENTED;
}

/* ==================== 音频 ==================== */

namespace {

bool load_wav(const char* path, int* sample_rate, std::vector<float>& data) {
    int32_t sr = 0;
    if (!load_wav_file(path, &sr, data)) return false;
    *sample_rate = sr;
    return true;
}

} // namespace

MDStatus md_audio_asr_wav(MDModelHandle h, const char* wav_path, const char** text) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !wav_path || !text) return MD_ERR_NULL_POINTER;
    if (!mh->ready || mh->kind != MD_MODEL_ASR) return MD_ERR_INVALID_ARGUMENT;
#ifdef BUILD_AUDIO
    std::vector<float> data;
    int sr = 0;
    if (!load_wav(wav_path, &sr, data)) { set_error_fmt("md_audio_asr_wav: cannot decode '%s'", wav_path); return MD_ERR_AUDIO_DECODE; }
    auto* m = static_cast<audio::asr::SenseVoice*>(mh->model);
    if (!m->predict(data, &mh->text_buf)) { set_error("md_audio_asr_wav: asr predict failed"); return MD_ERR_MODEL_PREDICT; }
    *text = mh->text_buf.c_str();
    return MD_OK;
#else
    (void)wav_path; (void)text;
    set_error("md_audio_asr_wav: built without BUILD_AUDIO");
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_asr(MDModelHandle h, const float* samples, size_t n, int sample_rate,
                      const char** text) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !samples || !text) return MD_ERR_NULL_POINTER;
    if (!mh->ready || mh->kind != MD_MODEL_ASR) return MD_ERR_INVALID_ARGUMENT;
    (void)sample_rate;
#ifdef BUILD_AUDIO
    std::vector<float> data(samples, samples + n);
    auto* m = static_cast<audio::asr::SenseVoice*>(mh->model);
    if (!m->predict(data, &mh->text_buf)) { set_error("md_audio_asr: asr predict failed"); return MD_ERR_MODEL_PREDICT; }
    *text = mh->text_buf.c_str();
    return MD_OK;
#else
    set_error("md_audio_asr: built without BUILD_AUDIO");
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_tts(MDModelHandle h, const char* text, const char* voice, float speed,
                      int* sample_rate, const float** audio, size_t* audio_n) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !text || !voice || !audio || !audio_n) return MD_ERR_NULL_POINTER;
    if (!mh->ready || mh->kind != MD_MODEL_TTS) return MD_ERR_INVALID_ARGUMENT;
#ifdef BUILD_AUDIO
    auto* m = static_cast<audio::tts::Kokoro*>(mh->model);
    if (!m->predict(text, voice, speed, &mh->audio_buf)) { set_error("md_audio_tts: tts predict failed"); return MD_ERR_MODEL_PREDICT; }
    if (sample_rate) *sample_rate = m->get_sample_rate();
    *audio = mh->audio_buf.data();
    *audio_n = mh->audio_buf.size();
    return MD_OK;
#else
    (void)text; (void)voice; (void)speed; (void)sample_rate; (void)audio; (void)audio_n;
    set_error("md_audio_tts: built without BUILD_AUDIO");
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

/* ==================== wav 落盘（独立实现，避免依赖 audio 模块） ==================== */

namespace {

void append_u32(std::vector<unsigned char>& b, uint32_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xff));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
    b.push_back(static_cast<unsigned char>((v >> 16) & 0xff));
    b.push_back(static_cast<unsigned char>((v >> 24) & 0xff));
}

void append_u16(std::vector<unsigned char>& b, uint16_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xff));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}

} // namespace

MDStatus md_wav_save(const float* samples, size_t n, int sample_rate, const char* path) {
    if (!samples || !path || !*path) return MD_ERR_NULL_POINTER;
    if (n == 0 || sample_rate <= 0) return MD_ERR_INVALID_ARGUMENT;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * 2;
    const uint32_t data_bytes = static_cast<uint32_t>(n) * 2;
    std::vector<unsigned char> wav;
    wav.reserve(44 + data_bytes);
    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    append_u32(wav, 36 + data_bytes);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    append_u32(wav, 16);
    append_u16(wav, 1);
    append_u16(wav, 1);
    append_u32(wav, static_cast<uint32_t>(sample_rate));
    append_u32(wav, byte_rate);
    append_u16(wav, 2);
    append_u16(wav, 16);
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(wav, data_bytes);
    for (size_t i = 0; i < n; ++i) {
        float s = samples[i];
        if (s > 1.f) s = 1.f;
        if (s < -1.f) s = -1.f;
        const int16_t v = static_cast<int16_t>(s * 32767.f);
        append_u16(wav, static_cast<uint16_t>(v));
    }
    FILE* f = std::fopen(path, "wb");
    if (!f) { set_error_fmt("md_wav_save: cannot open '%s'", path); return MD_ERR_INVALID_ARGUMENT; }
    const size_t written = std::fwrite(wav.data(), 1, wav.size(), f);
    std::fclose(f);
    if (written != wav.size()) return MD_ERR_INVALID_ARGUMENT;
    return MD_OK;
}

/* ==================== 结果 getter（数组式） ==================== */

MDStatus md_result_detection(MDResultHandle h, const MDDetectionItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_DETECTION) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<DetectionResult>*>(rh->data);
    auto* p = new ProjectedResult<MDDetectionItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDDetectionItem it{};
        it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
        it.score = r.score; it.label_id = r.label_id;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_classification(MDResultHandle h, const MDClassifyItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_CLASSIFICATION) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<ClassifyResult>*>(rh->data);
    auto* p = new ProjectedResult<MDClassifyItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        const size_t m = std::min(r.label_ids.size(), r.scores.size());
        for (size_t i = 0; i < m; ++i) {
            MDClassifyItem it{};
            it.label_id = r.label_ids[i];
            it.score = r.scores[i];
            p->v.push_back(it);
        }
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_pose(MDResultHandle h, const MDPoseItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<KeyPointsResult>*>(rh->data);
    auto* p = new ProjectedResult<MDPoseItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDPoseItem it{};
        it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
        it.score = r.score;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_keypoints(MDResultHandle h, size_t i, const MDPoint3** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDPoseItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<KeyPointsResult>*>(p->origin);
    if (kps) *kps = reinterpret_cast<const MDPoint3*>(origin->v[i].keypoints.data());
    if (n) *n = origin->v[i].keypoints.size();
    return MD_OK;
}

MDStatus md_result_obb(MDResultHandle h, const MDObbItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OBB) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<ObbResult>*>(rh->data);
    auto* p = new ProjectedResult<MDObbItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDObbItem it{};
        it.cx = r.rotated_box.xc; it.cy = r.rotated_box.yc;
        it.w = r.rotated_box.width; it.h = r.rotated_box.height;
        it.angle = r.rotated_box.angle;
        it.score = r.score; it.label_id = r.label_id;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_instance_seg(MDResultHandle h, const MDIsegItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<InstanceSegResult>*>(rh->data);
    auto* p = new ProjectedResult<MDIsegItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDIsegItem it{};
        it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
        it.score = r.score; it.label_id = r.label_id;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_mask(MDResultHandle h, size_t i, const unsigned char** buf, size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !buf) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDIsegItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<InstanceSegResult>*>(p->origin);
    const auto& r = origin->v[i];
    if (buf) *buf = r.mask.buffer.data();
    if (out_h) *out_h = r.mask.shape.empty() ? 0 : static_cast<size_t>(r.mask.shape[0]);
    if (out_w) *out_w = r.mask.shape.size() < 2 ? 0 : static_cast<size_t>(r.mask.shape[1]);
    return MD_OK;
}

MDStatus md_result_sem_seg(MDResultHandle h, const unsigned char** labels, size_t* out_h, size_t* out_w,
                           int* num_classes) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !labels) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_SEM_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<SingleResult<SemSegResult>*>(rh->data);
    if (labels) *labels = d->value.labels.data();
    if (out_h) *out_h = d->value.shape.empty() ? 0 : static_cast<size_t>(d->value.shape[0]);
    if (out_w) *out_w = d->value.shape.size() < 2 ? 0 : static_cast<size_t>(d->value.shape[1]);
    if (num_classes) *num_classes = d->value.num_classes;
    return MD_OK;
}

MDStatus md_result_depth(MDResultHandle h, const float** depth, size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !depth) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_DEPTH) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<SingleResult<DepthResult>*>(rh->data);
    if (depth) *depth = d->value.depth.data();
    if (out_h) *out_h = d->value.shape.empty() ? 0 : static_cast<size_t>(d->value.shape[0]);
    if (out_w) *out_w = d->value.shape.size() < 2 ? 0 : static_cast<size_t>(d->value.shape[1]);
    return MD_OK;
}

MDStatus md_result_face(MDResultHandle h, const MDFaceItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = new ProjectedResult<MDFaceItem>();
    if (auto* d = raw_result<KeyPointsResult>(rh); d) {
        p->v.reserve(d->v.size());
        for (const auto& f : d->v) {
            MDFaceItem it{};
            it.x = f.box.x; it.y = f.box.y; it.w = f.box.width; it.h = f.box.height;
            it.score = f.score;
            p->v.push_back(it);
        }
    } else if (auto* d = raw_result<face::InsightFaceBox>(rh); d) {
        p->v.reserve(d->v.size());
        for (const auto& f : d->v) {
            MDFaceItem it{};
            it.x = f.bbox[0]; it.y = f.bbox[1]; it.w = f.bbox[2] - f.bbox[0]; it.h = f.bbox[3] - f.bbox[1];
            it.score = f.score;
            p->v.push_back(it);
        }
    } else {
        delete p;
        return MD_ERR_INVALID_ARGUMENT;
    }
    p->origin = static_cast<ResultDataBase*>(rh->data);
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_face_kps(MDResultHandle h, size_t i, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDFaceItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    if (auto* origin = dynamic_cast<ResultData<KeyPointsResult>*>(p->origin)) {
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].keypoints.data());
        if (n) *n = origin->v[i].keypoints.size();
    } else if (auto* origin = dynamic_cast<ResultData<face::InsightFaceBox>*>(p->origin)) {
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].kps.data());
        if (n) *n = origin->v[i].kps.size();
    }
    return MD_OK;
}

MDStatus md_result_face_embedding(MDResultHandle h, size_t i,
                                  const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE_REC) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<FaceRecognitionResult>*>(rh->data);
    if (i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (emb_n) *emb_n = d->v[i].embedding.size();
    if (embedding) *embedding = d->v[i].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface(MDResultHandle h, const MDInsightFaceItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<face::InsightFaceResult>*>(rh->data);
    auto* p = new ProjectedResult<MDInsightFaceItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDInsightFaceItem it{};
        it.x = r.bbox[0]; it.y = r.bbox[1]; it.w = r.bbox[2] - r.bbox[0]; it.h = r.bbox[3] - r.bbox[1];
        it.score = r.det_score; it.gender = r.gender; it.age = r.age;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_insightface_kps(MDResultHandle h, size_t i, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDInsightFaceItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<face::InsightFaceResult>*>(p->origin);
    if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].kps.data());
    if (n) *n = origin->v[i].kps.size();
    return MD_OK;
}

MDStatus md_result_insightface_embedding(MDResultHandle h, size_t i,
                                         const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDInsightFaceItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<face::InsightFaceResult>*>(p->origin);
    if (emb_n) *emb_n = origin->v[i].embedding.size();
    if (embedding) *embedding = origin->v[i].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface_pose(MDResultHandle h, size_t i, const float** pose, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !pose || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDInsightFaceItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<face::InsightFaceResult>*>(p->origin);
    if (n) *n = origin->v[i].pose.size();
    if (pose && !origin->v[i].pose.empty()) *pose = origin->v[i].pose.data();
    return MD_OK;
}

MDStatus md_result_ocr(MDResultHandle h, size_t i, const int** quad, const char** text, float* score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<SingleResult<OCRResult>*>(rh->data);
    if (i >= d->value.boxes.size()) return MD_ERR_INVALID_ARGUMENT;
    if (quad) *quad = d->value.boxes[i].data();
    if (text) *text = i < d->value.text.size() ? d->value.text[i].c_str() : "";
    if (score) *score = i < d->value.rec_scores.size() ? d->value.rec_scores[i] : 0.f;
    return MD_OK;
}

MDStatus md_result_lpr(MDResultHandle h, const MDLprItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    auto* p = new ProjectedResult<MDLprItem>();
    if (auto* d = raw_result<LprResult>(rh); d) {
        p->v.reserve(d->v.size());
        for (const auto& r : d->v) {
            MDLprItem it{};
            it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
            it.score = r.score;
            p->v.push_back(it);
        }
    } else if (auto* d = raw_result<KeyPointsResult>(rh); d) {
        p->v.reserve(d->v.size());
        for (const auto& r : d->v) {
            MDLprItem it{};
            it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
            it.score = r.score;
            p->v.push_back(it);
        }
    } else {
        delete p;
        return MD_ERR_INVALID_ARGUMENT;
    }
    p->origin = static_cast<ResultDataBase*>(rh->data);
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_plate(MDResultHandle h, size_t i, const char** plate, const char** color) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !plate) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDLprItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    if (auto* origin = dynamic_cast<ResultData<LprResult>*>(p->origin)) {
        if (plate) *plate = origin->v[i].car_plate_str.c_str();
        if (color) *color = origin->v[i].car_plate_color.c_str();
    } else {
        if (plate) *plate = "";
        if (color) *color = "";
    }
    return MD_OK;
}

MDStatus md_result_ocr_cls(MDResultHandle h, size_t i, int* cls_label, float* cls_score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<SingleResult<OCRResult>*>(rh->data);
    if (cls_label) *cls_label = i < d->value.cls_labels.size() ? d->value.cls_labels[i] : 0;
    if (cls_score) *cls_score = i < d->value.cls_scores.size() ? d->value.cls_scores[i] : 0.f;
    return MD_OK;
}

MDStatus md_result_lpr_keypoints(MDResultHandle h, size_t i, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDLprItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    if (auto* origin = dynamic_cast<ResultData<LprResult>*>(p->origin)) {
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].keypoints.data());
        if (n) *n = origin->v[i].keypoints.size();
    } else {
        if (kps) *kps = nullptr;
        if (n) *n = 0;
    }
    return MD_OK;
}

MDStatus md_result_attribute(MDResultHandle h, const MDAttrItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = static_cast<ResultData<AttributeResult>*>(rh->data);
    auto* p = new ProjectedResult<MDAttrItem>();
    p->v.reserve(d->v.size());
    for (const auto& r : d->v) {
        MDAttrItem it{};
        it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
        it.box_score = r.box_score; it.box_label_id = r.box_label_id;
        p->v.push_back(it);
    }
    p->origin = d;
    rh->data = p;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_attr_scores(MDResultHandle h, size_t i, const float** scores, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !scores || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* p = static_cast<ProjectedResult<MDAttrItem>*>(rh->data);
    if (i >= p->count() || !p->origin) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = static_cast<ResultData<AttributeResult>*>(p->origin);
    if (n) *n = origin->v[i].attr_scores.size();
    if (scores) *scores = origin->v[i].attr_scores.data();
    return MD_OK;
}

MDStatus md_result_age(MDResultHandle h, int* age) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_AGE) return MD_ERR_INVALID_ARGUMENT;
    if (age) *age = static_cast<SingleResult<int>*>(rh->data)->value;
    return MD_OK;
}

MDStatus md_result_gender(MDResultHandle h, int* gender) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_GENDER) return MD_ERR_INVALID_ARGUMENT;
    if (gender) *gender = static_cast<SingleResult<int>*>(rh->data)->value;
    return MD_OK;
}

/* ==================== 绘制 ==================== */

namespace {

cv::Scalar md_color_to_scalar(MDColorRGBA c) {
    return cv::Scalar(c.b, c.g, c.r, c.a);
}

} // namespace

MDStatus md_draw_rect(MDImageHandle img, float x, float y, float w, float h,
                      MDColorRGBA color, float alpha) {
    auto* hi = static_cast<md_image_handle*>(img);
    if (!hi) return MD_ERR_NULL_POINTER;
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    const cv::Scalar cv_color = md_color_to_scalar(color);
    // 直接复用 C++ 的绘制实现（与 vis_* 系一致的 alpha 混合）
    modeldeploy::vision::draw_filled_rect(mat, {cvRound(x), cvRound(y), cvRound(w), cvRound(h)},
                                          cv_color, alpha);
    return MD_OK;
}

MDStatus md_draw_polygon(MDImageHandle img, const float* xs, const float* ys, size_t n,
                         MDColorRGBA color, float alpha) {
    auto* hi = static_cast<md_image_handle*>(img);
    if (!hi || !xs || !ys) return MD_ERR_NULL_POINTER;
    if (n < 3) return MD_ERR_INVALID_ARGUMENT;
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    std::vector<cv::Point> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) pts.emplace_back(cvRound(xs[i]), cvRound(ys[i]));
    const cv::Scalar cv_color = md_color_to_scalar(color);
    // 直接复用 C++ 的绘制实现（与 vis_ocr/vis_obb 一致的 fillPoly + addWeighted + polylines）
    modeldeploy::vision::draw_filled_polygon(mat, pts, cv_color, alpha);
    return MD_OK;
}

MDStatus md_draw_text(MDImageHandle img, float x, float y, const char* text,
                      const char* font_path, int font_size, MDColorRGBA color, float alpha) {
    auto* hi = static_cast<md_image_handle*>(img);
    if (!hi || !text) return MD_ERR_NULL_POINTER;
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    const cv::Scalar cv_color = md_color_to_scalar(color);
    (void)alpha;
    try {
        // 直接复用 C++ 的绘制实现（字体文件 / 内置字体统一走 C++ draw_text）
        modeldeploy::vision::draw_text(mat, text, font_path ? font_path : "",
                                       font_size, cv_color, {cvRound(x), cvRound(y)});
    } catch (...) {
        set_error("md_draw_text: text rendering failed");
        return MD_ERR_INVALID_ARGUMENT;
    }
    return MD_OK;
}

/* ==================== 结果可视化（复用 C++ vis_* 系列） ==================== */

namespace {

std::unordered_map<int, std::string> build_label_map(const MDLabelItem* items, size_t n) {
    std::unordered_map<int, std::string> m;
    if (!items) return m;
    for (size_t i = 0; i < n; ++i) {
        if (items[i].name) m[items[i].id] = items[i].name;
    }
    return m;
}

MDDrawOptions default_draw_options() {
    MDDrawOptions opt{};
    opt.threshold = 0.5;
    opt.font_size = 14;
    opt.alpha = 0.15;
    opt.save_result = 0;
    return opt;
}

// 就地绘制：把 ImageHandle 包成 ImageData（浅拷贝共享底层），vis_* 会写回
ImageData image_handle_as_data(md_image_handle* hi) {
    cv::Mat mat(hi->height, hi->width, CV_8UC3, hi->data);
    return ImageData(mat);
}

} // namespace

MDStatus md_draw_result(MDImageHandle img, MDResultHandle res, const MDDrawOptions* opt_in) {
    auto* hi = static_cast<md_image_handle*>(img);
    auto* rh = static_cast<md_result_handle*>(res);
    if (!hi || !rh) return MD_ERR_NULL_POINTER;

    const MDDrawOptions opt = opt_in ? *opt_in : default_draw_options();
    const double threshold = opt.threshold > 0 ? opt.threshold : 0.5;
    const std::string font_path = opt.font_path ? opt.font_path : "";
    const int font_size = opt.font_size > 0 ? opt.font_size : 14;
    const double alpha = opt.alpha > 0 ? opt.alpha : 0.15;
    const bool save = opt.save_result != 0;
    const auto label_map = build_label_map(opt.label_map, opt.label_map_size);

    ImageData image = image_handle_as_data(hi);

    try {
        switch (rh->kind) {
            case MD_RES_DETECTION: {
                auto* d = raw_result<DetectionResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_det(image, d->v, threshold, label_map, font_path, font_size, alpha, save);
                return MD_OK;
            }
            case MD_RES_OBB: {
                auto* d = raw_result<ObbResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_obb(image, d->v, threshold, font_path, font_size, alpha, save);
                return MD_OK;
            }
            case MD_RES_POSE: {
                auto* d = raw_result<KeyPointsResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_pose(image, d->v, font_path, font_size, 4, alpha, save);
                return MD_OK;
            }
            case MD_RES_FACE: {
                // face-det（Scrfd）结果为 KeyPointsResult；画框 + 关键点
                auto* d = raw_result<KeyPointsResult>(rh);
                if (d) {
                    modeldeploy::vision::vis_keypoints(image, d->v, font_path, font_size, 3, alpha, save, false);
                    return MD_OK;
                }
                set_error("md_draw_result: face result type unsupported");
                return MD_ERR_INVALID_ARGUMENT;
            }
            case MD_RES_INSIGHTFACE: {
                auto* d = raw_result<face::InsightFaceResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                std::vector<KeyPointsResult> kpr;
                kpr.reserve(d->v.size());
                for (const auto& r : d->v) {
                    KeyPointsResult kp;
                    kp.box = Rect2f{r.bbox[0], r.bbox[1], r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]};
                    kp.score = r.det_score;
                    for (const auto& p : r.kps) kp.keypoints.emplace_back(p[0], p[1], 0.f);
                    kpr.push_back(std::move(kp));
                }
                modeldeploy::vision::vis_keypoints(image, kpr, font_path, font_size, 3, alpha, save, false);
                return MD_OK;
            }
            case MD_RES_INSTANCE_SEG: {
                auto* d = raw_result<InstanceSegResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_iseg(image, d->v, threshold, font_path, font_size, alpha, save);
                return MD_OK;
            }
            case MD_RES_SEM_SEG: {
                auto* d = dynamic_cast<SingleResult<SemSegResult>*>(static_cast<ResultDataBase*>(rh->data));
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_sem(image, d->value, label_map, alpha, save);
                return MD_OK;
            }
            case MD_RES_DEPTH: {
                auto* d = dynamic_cast<SingleResult<DepthResult>*>(static_cast<ResultDataBase*>(rh->data));
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_depth(image, d->value, true, save);
                return MD_OK;
            }
            case MD_RES_OCR: {
                auto* d = dynamic_cast<SingleResult<OCRResult>*>(static_cast<ResultDataBase*>(rh->data));
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_ocr(image, d->value, font_path, font_size, alpha, save);
                return MD_OK;
            }
            case MD_RES_LPR: {
                auto* d = raw_result<LprResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_lpr(image, d->v, font_path, font_size, 4, alpha, save);
                return MD_OK;
            }
            case MD_RES_ATTR: {
                auto* d = raw_result<AttributeResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                modeldeploy::vision::vis_attr(image, d->v, threshold, label_map, font_path, font_size, alpha, save);
                return MD_OK;
            }
            case MD_RES_CLASSIFICATION: {
                auto* d = raw_result<ClassifyResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                if (!d->v.empty()) {
                    modeldeploy::vision::vis_cls(image, d->v[0], 1, threshold, font_path, font_size, alpha, save);
                }
                return MD_OK;
            }
            default:
                set_error_fmt("md_draw_result: unsupported result kind %d", (int)rh->kind);
                return MD_ERR_UNSUPPORTED_TYPE;
        }
    } catch (const std::exception& e) {
        set_error_fmt("md_draw_result: %s", e.what());
        return MD_ERR_INVALID_ARGUMENT;
    }
}