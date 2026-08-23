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
#include "csrc/vision/processors/processor_factory.h"
#include "csrc/vision/processors/cpu/cpu_processor_backend.h"
#ifdef WITH_GPU
#include "csrc/vision/processors/cuda/cuda_processor_backend.h"
#endif
#ifdef ENABLE_SOPHGO
#include "csrc/vision/processors/sophgo/sophgo_processor_backend.h"
#endif
#include "csrc/vision/face/insightface/face_analysis.h"
#include "csrc/vision/face/insightface/insightface_types.h"
#include "csrc/vision/ocr/ppocr.h"
#include "csrc/vision/ocr/dbdetector.h"
#include "csrc/vision/ocr/recognizer.h"
#include "csrc/vision/ocr/formula_recognition.h"
#include "csrc/vision/ocr/classifier.h"
#include "csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h"
#include "csrc/vision/lpr/lpr_det/lpr_det.h"
#include "csrc/vision/lpr/lpr_rec/lpr_rec.h"
#include "csrc/vision/barcode/barcode.h"
#include "csrc/vision/tracking/base_tracker.h"
#include "csrc/vision/tools/detections.h"
#include "csrc/vision/solutions/object_counter.h"
#include "csrc/vision/solutions/heatmap.h"
#include "csrc/vision/solutions/speed_estimator.h"
#include "csrc/vision/solutions/distance_estimator.h"
#include "csrc/vision/solutions/workout_monitor.h"
#include "csrc/vision/solutions/parking_manager.h"
#include "csrc/vision/tracking/bytetrack.h"
#include "csrc/vision/tracking/botsort.h"
#include "csrc/vision/tracking/strongsort.h"
#include "csrc/vision/pipeline/pedestrian_attribute.h"
#include "csrc/vision/common/visualize/utils.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/utils/wave_helper.h"
#include "csrc/utils/utils.h"
#include "csrc/core/md_log.h"
#include "csrc/vision/action/tsn.h"
#include "csrc/vision/action/st_gcn.h"
#include "csrc/vision/landmark/vehicle_keypoint.h"
#include "csrc/vision/landmark/face_landmark.h"

#ifdef BUILD_AUDIO
#include "csrc/audio/asr/sense_voice.h"
#include "csrc/audio/tts/kokoro.h"
#include "csrc/audio/speaker_verify/ecapa.h"
#include "csrc/audio/tools/resampler.h"
#include "csrc/audio/tools/audio_meta.h"
#include "csrc/audio/solutions/speaker_search.h"
#include "csrc/audio/solutions/tts_batcher.h"
#endif

/* ---------------- 句柄实现（全局命名空间，与 md_capi.h 前向声明对应） ---------------- */

/* 图像句柄：持有 BGR 数据（owns_data 时库持有）或引用外部内存 */
struct md_image_handle {
    bool owns_data = false;
    int width = 0;
    int height = 0;
    unsigned char* data = nullptr;
    modeldeploy::vision::ImageData image;  // 统一描述：CPU BGR（包 data）或设备 NV12 帧（零拷贝借用）
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
    std::shared_ptr<std::vector<float>> speaker_embed;  // 声纹 embedding 暂存（借用指针，随句柄存活）
    ~md_model_handle();
};

/* 结果句柄：统一容器，data 指向 ResultDataBase（T 由 kind 决定） */
struct md_result_handle {
    MDResultKind kind = MD_RES_DETECTION;
    void* data = nullptr;
    /* 批量逐图投影缓存：类型为 BatchProjectionBase*（首访时构建，析构释放），
       延迟把每图一组的结果投影为平铺 blittable 数组 + 每图偏移，返回稳定指针。 */
    void* batch_cache = nullptr;
    ~md_result_handle();
};

/* 跟踪器参数集：set_params 为位置参数且不可部分设置，故 C API 缓存完整参数集，
   每次命名 set_params 后按 kind 重建整组默认/已设值传给 C++ set_params。 */
struct TrackerParams {
    float track_thresh = 0.5f;
    float high_thresh = 0.5f;
    float low_thresh = 0.1f;
    int max_age = 30;
    int min_hits = 3;
    float iou_threshold = 0.3f;
    float match_thresh = 0.8f;
    float fuse_score_weight = 0.5f;
    float ema_alpha = 0.9f;
    float appearance_priority = 0.7f;
    bool with_cmc = true;
};

struct md_tracker_handle {
    MDTrackerKind kind = MD_TRACKER_BYTETRACK;
    std::unique_ptr<modeldeploy::vision::tracking::BaseTracker> tracker;
    TrackerParams params;
};

/* 条码识别器句柄：持有无状态 BarcodeDetector（纯 CV，detect 不推进状态） */
struct md_barcode_handle {
    modeldeploy::vision::barcode::BarcodeDetector det;
    uint32_t formats = modeldeploy::vision::barcode::FMT_ALL;
};

namespace {

/* ---------------- 错误模型（thread_local） ---------------- */
thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : "unknown error"; }
void set_error_fmt(const char* fmt, ...) {
    if (!fmt) { g_last_error = "unknown error"; return; }
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = vsnprintf(nullptr, 0, fmt, ap);  // 先测长度，避免 512 字节截断
    va_end(ap);
    if (n > 0) {
        std::string s(static_cast<size_t>(n), '\0');
        vsnprintf(s.data(), static_cast<size_t>(n) + 1, fmt, ap2);
        g_last_error = std::move(s);
    } else {
        g_last_error = "unknown error";
    }
    va_end(ap2);
}

/* 按 kind 把完整参数集派发给对应跟踪器的位置 set_params */
void apply_tracker_params(modeldeploy::vision::tracking::BaseTracker* t, MDTrackerKind kind,
                          const TrackerParams& p) {
    using namespace modeldeploy::vision::tracking;
    switch (kind) {
        case MD_TRACKER_BYTETRACK:
            static_cast<ByteTracker*>(t)->set_params(p.track_thresh, p.high_thresh, p.low_thresh,
                p.max_age, p.min_hits, p.iou_threshold);
            break;
        case MD_TRACKER_BOTSORT:
            static_cast<BotSortTracker*>(t)->set_params(p.track_thresh, p.high_thresh, p.low_thresh,
                p.max_age, p.min_hits, p.iou_threshold, p.match_thresh, p.fuse_score_weight,
                p.ema_alpha, p.with_cmc);
            break;
        case MD_TRACKER_STRONGSORT:
            static_cast<StrongSortTracker*>(t)->set_params(p.track_thresh, p.high_thresh, p.low_thresh,
                p.max_age, p.min_hits, p.iou_threshold, p.match_thresh, p.ema_alpha,
                p.appearance_priority, p.with_cmc);
            break;
        default:
            break;
    }
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

// 幂等投影：若 handle 已缓存同类型投影则直接复用；否则用 raw_result 解析出的真实
// origin 构建并替换。消除"同一数组 getter 二次调用把 ProjectedResult 当 ResultData 强转"的
// 未定义行为（旧实现 detection/classification 二次调用会读到野值甚至崩溃）。
template <typename Src, typename Dst, typename Fn>
ProjectedResult<Dst>* project_cached(md_result_handle* rh, Fn&& fill) {
    auto* base = static_cast<ResultDataBase*>(rh->data);
    if (auto* e = dynamic_cast<ProjectedResult<Dst>*>(base)) {
        return e;  // 已投影且类型相符 → 幂等复用，不再重投影/链式包裹
    }
    auto* d = raw_result<Src>(rh);
    if (!d) return nullptr;
    ProjectedResult<Dst>* p = new ProjectedResult<Dst>();
    fill(*p, d->v);   // 只填充投影项，无错误分支
    p->origin = d;    // 显式指向真实 origin
    rh->data = p;     // 最后提交
    return p;
}

// 原始（未投影）结果条数：始终读 origin，而非投影容器（投影数恒等于 origin，但语义应指原始结果）
size_t origin_count(md_result_handle* rh) {
    auto* base = static_cast<ResultDataBase*>(rh->data);
    if (auto* p = dynamic_cast<ProjectedResultBase*>(base)) {
        return p->origin_ptr()->count();
    }
    return base->count();
}

/* ---------------- 批量逐图投影（2D 批量结果 API） ----------------
 * 批量 md_model_predict_batch 现在把结果按图存储（变长 kind 为 ResultData<std::vector<T>>，
 * 整图/单值 kind 为 ResultData<T>）。逐图批量 getter 惰性把整批投影成
 * 「平铺 blittable 数组 + 每图偏移」，缓存在句柄 batch_cache 中；返回的每图指针
 * 在结果句柄存活期内稳定（无悬垂）。同一句柄只有一种 kind/一种 Dst 投影。 */
struct BatchProjectionBase {
    virtual ~BatchProjectionBase() = default;
};

template <typename Dst>
struct BatchProjection : BatchProjectionBase {
    std::vector<Dst> flat;       // 平铺：图0项 + 图1项 + ...
    std::vector<size_t> off;     // off.size()==images()+1；图 i 范围为 [off[i], off[i+1])
    template <typename Fill>
    explicit BatchProjection(size_t nimg, Fill&& fill) {
        off.push_back(0);
        for (size_t i = 0; i < nimg; ++i) { fill(flat, i); off.push_back(flat.size()); }
    }
};

// 惰性构建/复用某 kind 的逐图投影（同一句柄仅一种 Dst）
template <typename Dst, typename Fill>
BatchProjection<Dst>* ensure_batch_proj(md_result_handle* rh, size_t nimg, Fill&& fill) {
    if (auto* p = static_cast<BatchProjectionBase*>(rh->batch_cache))
        return static_cast<BatchProjection<Dst>*>(p);
    auto* p = new BatchProjection<Dst>(nimg, std::forward<Fill>(fill));
    rh->batch_cache = p;
    return p;
}

using namespace modeldeploy;
using namespace modeldeploy::vision;

/* 把 ImageHandle 转成 ImageData（零拷贝引用底层：CPU BGR 或设备 NV12） */
ImageData handle_to_image(const md_image_handle* hi) {
    return hi->image;
}

/* 是否为可直访问的 CPU BGR 数据（设备 NV12 帧 data==nullptr，无法做 CPU Mat 操作） */
inline bool handle_has_cpu_bgr(const md_image_handle* hi) {
    return hi && hi->data != nullptr && hi->image.device() == Device::CPU;
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
    MDDevice device = MD_DEV_CPU;
    int device_id = 0;
};

MDStatus md_option_create(MDOptionHandle* out) {
    if (!out) return MD_ERR_NULL_POINTER;
    *out = new md_option_handle();
    return MD_OK;
}

void md_option_destroy(MDOptionHandle h) {
    delete static_cast<md_option_handle*>(h);
}

void md_option_apply_device_(md_option_handle* o) {
    switch (o->device) {
        case MD_DEV_CPU: o->opt.use_cpu(); break;
        case MD_DEV_GPU: o->opt.use_gpu(o->device_id); break;
        case MD_DEV_TPU: o->opt.use_sophgo_backend(o->device_id); break;
        default:
            // OPENCL/VULKAN 为预留未实现枚举：明确报错，避免调用方误以为已启用
            set_error_fmt("md_option_set_device: device %d is reserved/not implemented", (int)o->device);
            break;
    }
}

void md_option_set_device(MDOptionHandle h, MDDevice d) {
    auto* o = static_cast<md_option_handle*>(h);
    o->device = d;
    md_option_apply_device_(o);
}

void md_option_set_device_id(MDOptionHandle h, int id) {
    auto* o = static_cast<md_option_handle*>(h);
    if (id < 0) id = 0;
    o->device_id = id;
    // 已设过 device 时立即生效，否则等 set_device 应用
    md_option_apply_device_(o);
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
    h->image = ImageData::from_raw(h->data, h->width, h->height, MdImageType::PKG_BGR_U8, false);
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
    h->image = ImageData::from_raw(h->data, h->width, h->height, MdImageType::PKG_BGR_U8, false);
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
    hi->image = ImageData::from_raw(hi->data, hi->width, hi->height, MdImageType::PKG_BGR_U8, false);
    *out = hi;
    return MD_OK;
}

static MDStatus image_from_image(MDImageHandle* out, ImageData&& img) {
    if (!out) return MD_ERR_NULL_POINTER;
    if (img.empty()) return MD_ERR_IMAGE_DECODE;
    const auto p0 = img.plane(0);
    if (!p0.data) return MD_ERR_IMAGE_DECODE;
    auto* h = new md_image_handle();
    h->width = img.width();
    h->height = img.height();
    const size_t bytes = static_cast<size_t>(h->width) * h->height * 3;
    h->data = new unsigned char[bytes];
    // 逐行拷贝，尊重源平面 step（CPU 紧致 BGR 步长==w*3，行为不变；padded-stride 帧不产出乱码）
    const size_t row_bytes = static_cast<size_t>(h->width) * 3;
    const int step_src = p0.step > 0 ? p0.step : static_cast<int>(row_bytes);
    for (int r = 0; r < h->height; ++r)
        std::memcpy(h->data + static_cast<size_t>(r) * row_bytes,
                    p0.data + static_cast<size_t>(r) * step_src, row_bytes);
    h->owns_data = true;
    h->image = ImageData::from_raw(h->data, h->width, h->height, MdImageType::PKG_BGR_U8, false);
    *out = h;
    return MD_OK;
}

MDStatus md_image_from_rgb24(MDImageHandle* out, const void* rgb, int w, int h) {
    if (!out || !rgb) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_rgb24: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    // 借调用方 RGB（零拷贝）→ cvt_color 产出自有 BGR 缓冲（不手造 cv::Mat）
    ImageData src = ImageData::from_raw(static_cast<unsigned char*>(const_cast<void*>(rgb)),
                                        w, h, MdImageType::PKG_RGB_U8, /*copy=*/false);
    if (src.empty()) {
        set_error("md_image_from_rgb24: invalid RGB buffer");
        return MD_ERR_IMAGE_DECODE;
    }
    ImageData bgr = ImageData::cvt_color(src, ColorConvertType::CVT_PA_RGB2PA_BGR);
    if (bgr.empty()) {
        const char* le = ImageData::last_error();
        set_error_fmt("md_image_from_rgb24: convert failed (%s)", (le && *le) ? le : "unknown");
        return MD_ERR_IMAGE_DECODE;
    }
    // from_raw(copy=false) 借用的调用方指针仅用于转 BGR；cvt_color 返回到自有缓冲，
    // 由 move 进句柄的 ImageData（impl 内 owner shared_ptr）持有 → 无悬垂。
    auto* hi = new md_image_handle();
    hi->width = w;
    hi->height = h;
    hi->data = const_cast<uint8_t*>(bgr.plane(0).data);  // 指向将被 image 拥有的缓冲
    hi->owns_data = false;
    hi->image = std::move(bgr);   // 句柄持有自有 BGR 缓冲
    *out = hi;
    return MD_OK;
}

MDStatus md_image_from_nv12(MDImageHandle* out, const void* y, const void* uv,
                            int w, int h, int step_y, int step_uv) {
    if (!out || !y) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_nv12: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    ImageData::Plane pl[2] = {
        {static_cast<const uint8_t*>(y), step_y > 0 ? step_y : w},
        {static_cast<const uint8_t*>(uv), step_uv > 0 ? step_uv : w},
    };
    auto img = ImageData::from_planes(pl, uv ? 2 : 1, MdImageType::NV12, w, h, Device::CPU);
    if (img.empty()) {
        set_error("md_image_from_nv12: failed to construct NV12");
        return MD_ERR_IMAGE_DECODE;
    }
    auto bgr = ImageData::cvt_color(img, ColorConvertType::CVT_NV122PKG_BGR);
    if (bgr.empty()) {
        const char* le = ImageData::last_error();
        set_error_fmt("md_image_from_nv12: convert failed (%s)", (le && *le) ? le : "unknown");
        return MD_ERR_IMAGE_DECODE;
    }
    return image_from_image(out, std::move(bgr));
}

/* 自有版 NV12：把调用方 Y/UV 拷入库内自有缓冲，产真 NV12 两平面帧（安全，无需调用方保活）。
   与 md_image_from_nv12（转 BGR 拷贝）不同：保留 NV12 类型，走零拷贝 NV12 推理路径。 */
MDStatus md_image_from_nv12_owned(MDImageHandle* out, const void* y, const void* uv,
                                  int w, int h, int step_y, int step_uv) {
    if (!out || !y) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_nv12_owned: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    if (step_y <= 0) step_y = w;
    if (step_uv <= 0) step_uv = w;
    const size_t y_bytes = static_cast<size_t>(step_y) * h;
    const size_t uv_bytes = static_cast<size_t>(step_uv) * (static_cast<size_t>(h) / 2);
    auto* mem = new uint8_t[y_bytes + uv_bytes];
    uint8_t* yown = mem;
    uint8_t* uvown = mem + y_bytes;
    std::memcpy(yown, y, y_bytes);
    if (uv) std::memcpy(uvown, uv, uv_bytes);
    auto owner = std::shared_ptr<void>(mem, [](void* p) { delete[] static_cast<uint8_t*>(p); });
    ImageData::Plane pl[2] = {{yown, step_y}, {uvown, step_uv}};
    auto img = ImageData::from_planes(pl, uv ? 2 : 1, MdImageType::NV12, w, h, Device::CPU, std::move(owner));
    if (img.empty()) {
        set_error("md_image_from_nv12_owned: failed to construct NV12");
        return MD_ERR_IMAGE_DECODE;
    }
    auto* hi = new md_image_handle();
    hi->width = w;
    hi->height = h;
    hi->data = nullptr;
    hi->owns_data = false;
    hi->image = std::move(img);  // 句柄经 owner 持有自有缓冲
    *out = hi;
    return MD_OK;
}

MDStatus md_image_from_device_nv12(MDImageHandle* out, const void* y, const void* uv,
                                    int w, int h, int step_y, int step_uv, MDDevice dev) {
    if (!out || !y) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_device_nv12: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    if (step_y <= 0) step_y = w;
    if (step_uv <= 0) step_uv = w;
    Device d;
    switch (dev) {
        case MD_DEV_GPU: d = Device::GPU; break;
        case MD_DEV_TPU: d = Device::TPU; break;
        default: d = Device::CPU; break;
    }
    ImageData::Plane pl[2] = {
        {static_cast<const uint8_t*>(y), step_y},
        {static_cast<const uint8_t*>(uv), step_uv},
    };
    auto img = ImageData::from_planes(pl, uv ? 2 : 1, MdImageType::NV12, w, h, d);
    if (img.empty()) {
        set_error("md_image_from_device_nv12: failed to construct device frame");
        return MD_ERR_INVALID_ARGUMENT;
    }
    auto* hi = new md_image_handle();
    hi->width = w;
    hi->height = h;
    hi->data = nullptr;
    hi->owns_data = false;
    hi->image = std::move(img);
    *out = hi;
    return MD_OK;
}

MDStatus md_image_from_yuv420p(MDImageHandle* out, const void* data, int w, int h) {
    if (!out || !data) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) { set_error("md_image_from_yuv420p: invalid size"); return MD_ERR_INVALID_ARGUMENT; }
    // 平铺 I420(Y[h*w], U[w*h/4], V[w*h/4]) → 三平面 from_planes → CVT_I4202PKG_BGR
    const auto* p = static_cast<const uint8_t*>(data);
    ImageData::Plane pl[3] = {
        {p, w},
        {p + static_cast<size_t>(w) * h, w / 2},
        {p + static_cast<size_t>(w) * h * 5 / 4, w / 2},
    };
    auto nv = ImageData::from_planes(pl, 3, MdImageType::I420, w, h, Device::CPU);
    if (nv.empty()) {
        set_error("md_image_from_yuv420p: failed to construct I420");
        return MD_ERR_IMAGE_DECODE;
    }
    auto bgr = ImageData::cvt_color(nv, ColorConvertType::CVT_I4202PKG_BGR);
    if (bgr.empty()) {
        const char* le = ImageData::last_error();
        set_error_fmt("md_image_from_yuv420p: convert failed (%s)", (le && *le) ? le : "unknown");
        return MD_ERR_IMAGE_DECODE;
    }
    return image_from_image(out, std::move(bgr));
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
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_image_clone: device frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
    auto* nh = new md_image_handle();
    nh->width = mat.cols;
    nh->height = mat.rows;
    const size_t bytes = static_cast<size_t>(mat.total()) * static_cast<size_t>(mat.elemSize());
    const size_t row_bytes = static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.elemSize());
    nh->data = new unsigned char[bytes];
    // 逐行拷贝，尊重 mat.step（CPU 紧致/连续步骤长时行为字节等价；padded-stride 的 CPU BGR 帧不产出乱码）
    for (int r = 0; r < mat.rows; ++r)
        std::memcpy(nh->data + static_cast<size_t>(r) * row_bytes,
                    mat.data + static_cast<size_t>(r) * mat.step, row_bytes);
    nh->owns_data = true;
    nh->image = ImageData::from_raw(nh->data, nh->width, nh->height, MdImageType::PKG_BGR_U8, false);
    *out = nh;
    return MD_OK;
}

MDStatus md_image_crop(MDImageHandle in, int x, int y, int w, int h, MDImageHandle* out) {
    if (!in || !out) return MD_ERR_NULL_POINTER;
    if (w <= 0 || h <= 0) return MD_ERR_INVALID_ARGUMENT;
    const auto* hi = static_cast<md_image_handle*>(in);
    if (!handle_has_cpu_bgr(hi)) { set_error("md_image_crop: device frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
    if (x < 0 || y < 0 || x + w > hi->width || y + h > hi->height) {
        set_error("md_image_crop: crop rect out of bounds");
        return MD_ERR_INVALID_ARGUMENT;
    }
    auto src = handle_to_image(hi);
    auto r = src.crop({static_cast<float>(x), static_cast<float>(y),
                       static_cast<float>(w), static_cast<float>(h)});
    if (r.empty()) {
        const char* le = ImageData::last_error();
        set_error_fmt("md_image_crop: %s", (le && *le) ? le : "crop failed");
        return MD_ERR_INVALID_ARGUMENT;
    }
    cv::Mat mat;
    if (!r.asMat(&mat) || mat.empty()) {
        set_error("md_image_crop: cannot take cropped mat");
        return MD_ERR_INVALID_ARGUMENT;
    }
    return image_from_mat(out, std::move(mat));
}

MDStatus md_image_show(MDImageHandle h) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi) return MD_ERR_NULL_POINTER;
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_image_show: device or non-packed frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
#ifdef HAVE_OPENCV_HIGHGUI
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
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_image_save: device or non-packed frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
    if (!cv::imwrite(path, mat)) { set_error_fmt("md_image_save: failed to write '%s'", path); return MD_ERR_INVALID_ARGUMENT; }
    return MD_OK;
}

MDStatus md_image_encode(MDImageHandle h, const char* ext,
                         const unsigned char** buf, size_t* n) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi || !ext || !buf || !n) return MD_ERR_NULL_POINTER;
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_image_encode: device or non-packed frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
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

MDStatus md_image_plane_ptrs(MDImageHandle h, MDDevice* dev, void** y, void** uv) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi || !dev || !y || !uv) return MD_ERR_NULL_POINTER;
    *y = nullptr; *uv = nullptr; *dev = MD_DEV_CPU;
    if (hi->image.type() != MdImageType::NV12) return MD_ERR_UNSUPPORTED_TYPE;
    switch (hi->image.device()) {
        case Device::GPU: *dev = MD_DEV_GPU; break;
        case Device::TPU: *dev = MD_DEV_TPU; break;
        default: *dev = MD_DEV_CPU; break;
    }
    *y = const_cast<uint8_t*>(hi->image.plane(0).data);
    *uv = const_cast<uint8_t*>(hi->image.plane(1).data);
    return (*y) ? MD_OK : MD_ERR_INVALID_ARGUMENT;
}

MDStatus md_image_info(MDImageHandle h, int* type, int* dev, int* nplanes) {
    auto* hi = static_cast<md_image_handle*>(h);
    if (!hi) return MD_ERR_NULL_POINTER;
    if (type)    *type    = static_cast<int>(hi->image.type());
    if (nplanes) *nplanes = static_cast<int>(hi->image.plane_count());
    if (dev) {
        // MDDevice 与 Device 数值映射（TPU/OPENCL/VULKAN 顺序不同，须 switch 对齐现有 md_image_plane_ptrs）
        switch (hi->image.device()) {
            case Device::GPU:    *dev = static_cast<int>(MD_DEV_GPU); break;
            case Device::TPU:    *dev = static_cast<int>(MD_DEV_TPU); break;
            case Device::OPENCL: *dev = static_cast<int>(MD_DEV_OPENCL); break;
            case Device::VULKAN: *dev = static_cast<int>(MD_DEV_VULKAN); break;
            default:             *dev = static_cast<int>(MD_DEV_CPU); break;
        }
    }
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
        case MD_MODEL_HAND: {
            mh->model = make_model<hand::HandKeypoint>(model_path, opt, "HandKeypoint", &err);
            if (!mh->model) return fail_init("HandKeypoint");
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
        case MD_MODEL_REID: {
            mh->model = make_model<reid::ReID>(model_path, opt, "ReID", &err);
            if (!mh->model) return fail_init("ReID");
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
            mh->model = make_model<face::SeetaFaceAsFirst>(model_path, opt, "SeetaFaceAsFirst", &err);
            if (!mh->model) return fail_init("SeetaFaceAsFirst");
            break;
        }
        case MD_MODEL_FACE_AS_SECOND: {
            mh->model = make_model<face::SeetaFaceAsSecond>(model_path, opt, "SeetaFaceAsSecond", &err);
            if (!mh->model) return fail_init("SeetaFaceAsSecond");
            break;
        }
        case MD_MODEL_FACE_AS_PIPELINE: {
            if (!need_parts(3, "face-as-pipeline")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new face::SeetaFaceAsPipeline(parts[0], parts[1], parts[2], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("SeetaFaceAsPipeline");
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
        case MD_MODEL_FORMULA_RECOGNIZER: {
            if (!need_parts(1, "formula-recognizer")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new ocr::FormulaRecognizer(parts[0], parts.size() > 1 ? parts[1] : "", opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("FormulaRecognizer");
            break;
        }
        case MD_MODEL_TSN: {
            if (!need_parts(1, "tsn")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new action::TSN(parts[0], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("TSN");
            break;
        }
        case MD_MODEL_ST_GCN: {
            if (!need_parts(1, "st-gcn")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new action::StGcn(parts[0], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("StGcn");
            break;
        }
        case MD_MODEL_VEHICLE_KEYPOINT: {
            mh->model = make_model<landmark::VehicleKeypoint>(model_path, opt, "VehicleKeypoint", &err);
            if (!mh->model) return fail_init("VehicleKeypoint");
            break;
        }
        case MD_MODEL_FACE_LANDMARK: {
            mh->model = make_model<landmark::FaceLandmark>(model_path, opt, "FaceLandmark", &err);
            if (!mh->model) return fail_init("FaceLandmark");
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
        case MD_MODEL_SPEAKER_VERIFY: {
            if (!need_parts(1, "speaker-verify")) return MD_ERR_INVALID_ARGUMENT;
            const auto parts = split_path(model_path);
            auto* m = new audio::speaker_verify::SpeakerVerify(parts[0], opt);
            mh->model = m;
            if (!m->is_initialized()) return fail_init("SpeakerVerify");
            break;
        }
#else
        case MD_MODEL_ASR:
        case MD_MODEL_TTS:
        case MD_MODEL_SPEAKER_VERIFY:
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
        case MD_MODEL_HAND: delete static_cast<hand::HandKeypoint*>(model); break;
        case MD_MODEL_OBB: delete static_cast<detection::UltralyticsObb*>(model); break;
        case MD_MODEL_INSTANCE_SEG: delete static_cast<detection::UltralyticsSeg*>(model); break;
        case MD_MODEL_SEM_SEG: delete static_cast<detection::UltralyticsSem*>(model); break;
        case MD_MODEL_DEPTH: delete static_cast<detection::UltralyticsDepth*>(model); break;
        case MD_MODEL_FACE_DET: delete static_cast<face::Scrfd*>(model); break;
        case MD_MODEL_FACE_REC: delete static_cast<face::SeetaFaceID*>(model); break;
        case MD_MODEL_REID: delete static_cast<reid::ReID*>(model); break;
        case MD_MODEL_FACE_AGE: delete static_cast<face::SeetaFaceAge*>(model); break;
        case MD_MODEL_FACE_GENDER: delete static_cast<face::SeetaFaceGender*>(model); break;
        case MD_MODEL_FACE_AS: delete static_cast<face::SeetaFaceAsFirst*>(model); break;
        case MD_MODEL_FACE_AS_SECOND: delete static_cast<face::SeetaFaceAsSecond*>(model); break;
        case MD_MODEL_FACE_AS_PIPELINE: delete static_cast<face::SeetaFaceAsPipeline*>(model); break;
        case MD_MODEL_FACE_REC_PIPELINE: delete static_cast<face::FaceRecognizerPipeline*>(model); break;
        case MD_MODEL_INSIGHTFACE: delete static_cast<face::InsightFaceAnalysis*>(model); break;
        case MD_MODEL_INSIGHTFACE_DET: delete static_cast<face::InsightFaceDet*>(model); break;
        case MD_MODEL_OCR: delete static_cast<ocr::PaddleOCR*>(model); break;
        case MD_MODEL_OCR_DET: delete static_cast<ocr::DBDetector*>(model); break;
        case MD_MODEL_OCR_REC: delete static_cast<ocr::Recognizer*>(model); break;
        case MD_MODEL_FORMULA_RECOGNIZER: delete static_cast<ocr::FormulaRecognizer*>(model); break;
        case MD_MODEL_TSN: delete static_cast<action::TSN*>(model); break;
        case MD_MODEL_ST_GCN: delete static_cast<action::StGcn*>(model); break;
        case MD_MODEL_VEHICLE_KEYPOINT: delete static_cast<landmark::VehicleKeypoint*>(model); break;
        case MD_MODEL_FACE_LANDMARK: delete static_cast<landmark::FaceLandmark*>(model); break;
        case MD_MODEL_OCR_CLS: delete static_cast<ocr::Classifier*>(model); break;
        case MD_MODEL_LPR_DET: delete static_cast<lpr::LprDetection*>(model); break;
        case MD_MODEL_LPR_REC: delete static_cast<lpr::LprRecognizer*>(model); break;
        case MD_MODEL_LPR_PIPELINE: delete static_cast<lpr::LprPipeline*>(model); break;
        case MD_MODEL_PED_ATTR: delete static_cast<pipeline::PedestrianAttribute*>(model); break;
#ifdef BUILD_AUDIO
        case MD_MODEL_ASR: delete static_cast<audio::asr::SenseVoice*>(model); break;
        case MD_MODEL_TTS: delete static_cast<audio::tts::Kokoro*>(model); break;
        case MD_MODEL_SPEAKER_VERIFY: delete static_cast<audio::speaker_verify::SpeakerVerify*>(model); break;
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
        case MD_MODEL_HAND: cloned = static_cast<hand::HandKeypoint*>(src->model)->clone().release(); break;
        case MD_MODEL_OBB: cloned = static_cast<detection::UltralyticsObb*>(src->model)->clone().release(); break;
        case MD_MODEL_INSTANCE_SEG: cloned = static_cast<detection::UltralyticsSeg*>(src->model)->clone().release(); break;
        case MD_MODEL_SEM_SEG: cloned = static_cast<detection::UltralyticsSem*>(src->model)->clone().release(); break;
        case MD_MODEL_DEPTH: cloned = static_cast<detection::UltralyticsDepth*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_DET: cloned = static_cast<face::Scrfd*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_REC: cloned = static_cast<face::SeetaFaceID*>(src->model)->clone().release(); break;
        case MD_MODEL_REID: cloned = static_cast<reid::ReID*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AGE: cloned = static_cast<face::SeetaFaceAge*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_GENDER: cloned = static_cast<face::SeetaFaceGender*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_REC_PIPELINE: cloned = static_cast<face::FaceRecognizerPipeline*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AS: cloned = static_cast<face::SeetaFaceAsFirst*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AS_SECOND: cloned = static_cast<face::SeetaFaceAsSecond*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_AS_PIPELINE: cloned = static_cast<face::SeetaFaceAsPipeline*>(src->model)->clone().release(); break;
        case MD_MODEL_INSIGHTFACE_DET: cloned = static_cast<face::InsightFaceDet*>(src->model)->clone().release(); break;
        case MD_MODEL_INSIGHTFACE: cloned = static_cast<face::InsightFaceAnalysis*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR: cloned = static_cast<ocr::PaddleOCR*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_DET: cloned = static_cast<ocr::DBDetector*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_REC: cloned = static_cast<ocr::Recognizer*>(src->model)->clone().release(); break;
        case MD_MODEL_FORMULA_RECOGNIZER: cloned = static_cast<ocr::FormulaRecognizer*>(src->model)->clone().release(); break;
        case MD_MODEL_TSN: cloned = static_cast<action::TSN*>(src->model)->clone().release(); break;
        case MD_MODEL_ST_GCN: cloned = static_cast<action::StGcn*>(src->model)->clone().release(); break;
        case MD_MODEL_VEHICLE_KEYPOINT: cloned = static_cast<landmark::VehicleKeypoint*>(src->model)->clone().release(); break;
        case MD_MODEL_FACE_LANDMARK: cloned = static_cast<landmark::FaceLandmark*>(src->model)->clone().release(); break;
        case MD_MODEL_OCR_CLS: cloned = static_cast<ocr::Classifier*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_DET: cloned = static_cast<lpr::LprDetection*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_REC: cloned = static_cast<lpr::LprRecognizer*>(src->model)->clone().release(); break;
        case MD_MODEL_LPR_PIPELINE: cloned = static_cast<lpr::LprPipeline*>(src->model)->clone().release(); break;
        case MD_MODEL_PED_ATTR: cloned = static_cast<pipeline::PedestrianAttribute*>(src->model)->clone().release(); break;
#ifdef BUILD_AUDIO
        case MD_MODEL_ASR: cloned = static_cast<audio::asr::SenseVoice*>(src->model)->clone().release(); break;
        case MD_MODEL_TTS: cloned = static_cast<audio::tts::Kokoro*>(src->model)->clone().release(); break;
        case MD_MODEL_SPEAKER_VERIFY: cloned = static_cast<audio::speaker_verify::SpeakerVerify*>(src->model)->clone().release(); break;
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
        case MD_MODEL_HAND: static_cast<hand::HandKeypoint*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_VEHICLE_KEYPOINT: static_cast<landmark::VehicleKeypoint*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_OBB: static_cast<detection::UltralyticsObb*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_INSTANCE_SEG: static_cast<detection::UltralyticsSeg*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_SEM_SEG: static_cast<detection::UltralyticsSem*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_DEPTH: static_cast<detection::UltralyticsDepth*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_FACE_DET: static_cast<face::Scrfd*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_PED_ATTR: static_cast<pipeline::PedestrianAttribute*>(mh->model)->set_det_input_size(size); break;
        case MD_MODEL_LPR_DET: static_cast<lpr::LprDetection*>(mh->model)->get_preprocessor().set_size(size); break;
        case MD_MODEL_FACE_REC_PIPELINE:
            static_cast<face::FaceRecognizerPipeline*>(mh->model)->get_detector()->get_preprocessor().set_size(size);
            break;
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

MDStatus md_model_set_cls_batch_size(MDModelHandle handle, int batch) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (batch == 0 || batch < -1) {
        set_error_fmt("md_model_set_cls_batch_size: invalid batch %d (must be >0 or -1)", batch);
        return MD_ERR_INVALID_ARGUMENT;
    }
    switch (mh->kind) {
        case MD_MODEL_PED_ATTR:
            if (!static_cast<pipeline::PedestrianAttribute*>(mh->model)->set_cls_batch_size(batch)) {
                set_error("md_model_set_cls_batch_size: set_cls_batch_size rejected value");
                return MD_ERR_INVALID_ARGUMENT;
            }
            break;
        case MD_MODEL_OCR:
            if (!static_cast<ocr::PaddleOCR*>(mh->model)->set_cls_batch_size(batch)) {
                set_error("md_model_set_cls_batch_size: set_cls_batch_size rejected value");
                return MD_ERR_INVALID_ARGUMENT;
            }
            break;
        default:
            set_error_fmt("md_model_set_cls_batch_size: unsupported for kind %d", (int)mh->kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

MDStatus md_model_set_rec_batch_size(MDModelHandle handle, int batch) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (batch == 0 || batch < -1) {
        set_error_fmt("md_model_set_rec_batch_size: invalid batch %d (must be >0 or -1)", batch);
        return MD_ERR_INVALID_ARGUMENT;
    }
    switch (mh->kind) {
        case MD_MODEL_OCR:
            if (!static_cast<ocr::PaddleOCR*>(mh->model)->set_rec_batch_size(batch)) {
                set_error("md_model_set_rec_batch_size: set_rec_batch_size rejected value");
                return MD_ERR_INVALID_ARGUMENT;
            }
            break;
        default:
            set_error_fmt("md_model_set_rec_batch_size: unsupported for kind %d", (int)mh->kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

MDStatus md_model_set_rec_image_shape(MDModelHandle handle, int c, int h, int w) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (c <= 0 || h <= 0 || w <= 0) return MD_ERR_INVALID_ARGUMENT;
    const std::vector<int> shape{c, h, w};
    switch (mh->kind) {
        case MD_MODEL_OCR_REC:
            static_cast<ocr::Recognizer*>(mh->model)->get_preprocessor().set_rec_image_shape(shape);
            break;
        case MD_MODEL_OCR:
            static_cast<ocr::PaddleOCR*>(mh->model)->get_recognizer()->get_preprocessor().set_rec_image_shape(shape);
            break;
        default:
            set_error_fmt("md_model_set_rec_image_shape: unsupported for kind %d", (int)mh->kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

/* ==================== 模型前/后处理参数（扁平参数名分发表） ==================== */

namespace {

// 参数类型标记（自省返回字符）
enum ParamType { PT_I = 'I', PT_D = 'D', PT_B = 'B', PT_S = 'S' };

// 依据 kind 上报支持的参数名列表（'|' 拼接，静态）
const char* kind_param_names(MDModelKind kind) {
    switch (kind) {
        case MD_MODEL_DETECTION:
        case MD_MODEL_OBB:
            return "conf_threshold|nms_threshold";
        case MD_MODEL_POSE:
        case MD_MODEL_HAND:
        case MD_MODEL_VEHICLE_KEYPOINT:
            return "conf_threshold|nms_threshold|keypoints_num";
        case MD_MODEL_INSTANCE_SEG:
            return "conf_threshold|nms_threshold|mask_threshold";
        case MD_MODEL_CLASSIFICATION:
            return "top_k|multi_label";
        case MD_MODEL_FACE_DET:
        case MD_MODEL_FACE_REC_PIPELINE:
            return "conf_threshold|nms_threshold|landmarks_per_face";
        case MD_MODEL_OCR_DET:
            return "det_db_thresh|det_db_box_thresh|det_db_unclip_ratio|det_db_score_mode|use_dilation|max_side_len";
        case MD_MODEL_OCR:
            return "det_db_thresh|det_db_box_thresh|det_db_unclip_ratio|det_db_score_mode|use_dilation|cls_thresh|max_side_len";
        case MD_MODEL_OCR_CLS:
            return "cls_thresh";
        case MD_MODEL_PED_ATTR:
            return "det_threshold";
        case MD_MODEL_INSIGHTFACE:
            return "det_thresh";
        case MD_MODEL_LPR_DET:
            return "conf_threshold|nms_threshold|landmarks_per_card";
        default:
            return "";
    }
}

// 依据 kind 上报某参数的类型字符（'I'/'D'/'B'/'S'），未知返回 0
char param_type_of(MDModelKind kind, const char* name) {
    if (!name) return 0;
    const bool is_det = std::strcmp(name, "conf_threshold") == 0 || std::strcmp(name, "nms_threshold") == 0;
    switch (kind) {
        case MD_MODEL_DETECTION:
        case MD_MODEL_POSE:
        case MD_MODEL_HAND:
        case MD_MODEL_VEHICLE_KEYPOINT:
        case MD_MODEL_OBB:
        case MD_MODEL_INSTANCE_SEG:
            if (is_det) return PT_D;
            if ((kind == MD_MODEL_POSE || kind == MD_MODEL_HAND || kind == MD_MODEL_VEHICLE_KEYPOINT) &&
                std::strcmp(name, "keypoints_num") == 0) return PT_I;
            if (kind == MD_MODEL_INSTANCE_SEG && std::strcmp(name, "mask_threshold") == 0) return PT_D;
            return 0;
        case MD_MODEL_CLASSIFICATION:
            if (std::strcmp(name, "top_k") == 0) return PT_I;
            if (std::strcmp(name, "multi_label") == 0) return PT_B;
            return 0;
        case MD_MODEL_FACE_DET:
        case MD_MODEL_FACE_REC_PIPELINE:
            if (is_det) return PT_D;
            if (std::strcmp(name, "landmarks_per_face") == 0) return PT_I;
            return 0;
        case MD_MODEL_OCR_DET:
            if (std::strcmp(name, "det_db_thresh") == 0) return PT_D;
            if (std::strcmp(name, "det_db_box_thresh") == 0) return PT_D;
            if (std::strcmp(name, "det_db_unclip_ratio") == 0) return PT_D;
            if (std::strcmp(name, "det_db_score_mode") == 0) return PT_S;
            if (std::strcmp(name, "use_dilation") == 0) return PT_B;
            if (std::strcmp(name, "max_side_len") == 0) return PT_I;
            return 0;
        case MD_MODEL_OCR:
            if (std::strcmp(name, "det_db_thresh") == 0) return PT_D;
            if (std::strcmp(name, "det_db_box_thresh") == 0) return PT_D;
            if (std::strcmp(name, "det_db_unclip_ratio") == 0) return PT_D;
            if (std::strcmp(name, "det_db_score_mode") == 0) return PT_S;
            if (std::strcmp(name, "use_dilation") == 0) return PT_B;
            if (std::strcmp(name, "cls_thresh") == 0) return PT_D;
            if (std::strcmp(name, "max_side_len") == 0) return PT_I;
            return 0;
        case MD_MODEL_OCR_CLS:
            if (std::strcmp(name, "cls_thresh") == 0) return PT_D;
            return 0;
        case MD_MODEL_PED_ATTR:
            if (std::strcmp(name, "det_threshold") == 0) return PT_D;
            return 0;
        case MD_MODEL_LPR_DET:
            if (std::strcmp(name, "conf_threshold") == 0 || std::strcmp(name, "nms_threshold") == 0 ||
                std::strcmp(name, "landmarks_per_card") == 0) return PT_D;
            return 0;
        case MD_MODEL_INSIGHTFACE:
            if (std::strcmp(name, "det_thresh") == 0) return PT_D;
            return 0;
        default:
            return 0;
    }
}

// 按 kind+参数名分派到具体模型 setter。返回 0=ok，否则返回对应 MD_ERR_* 并已 set_error。
int apply_model_param(md_model_handle* mh, const char* name, char req_type,
                      int64_t i, double d, const char* s) {
    const MDModelKind kind = mh->kind;
    const void* m = mh->model;
    const char decl_type = param_type_of(kind, name);
    if (decl_type == 0) {
        const char* supported = kind_param_names(kind);
        if (!*supported)
            set_error_fmt("md_model_set_param: kind %d has no supported params", (int)kind);
        else
            set_error_fmt("md_model_set_param: unknown param '%s' for kind %d (supported: %s)",
                          name, (int)kind, supported);
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (decl_type != req_type) {
        set_error_fmt("md_model_set_param: param '%s' expects type '%c' but got '%c'",
                      name, decl_type, req_type);
        return MD_ERR_INVALID_TYPE;
    }

    const bool is_bool = req_type == PT_B;
    const bool enable = is_bool && i != 0;

    switch (kind) {
        case MD_MODEL_DETECTION: {
            auto* pm = static_cast<detection::UltralyticsDet*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else pm->get_postprocessor().set_nms_threshold((float)d);
            break;
        }
        case MD_MODEL_POSE: {
            auto* pm = static_cast<detection::UltralyticsPose*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
            else pm->get_postprocessor().set_keypoints_num((int)i);
            break;
        }
        case MD_MODEL_HAND: {
            auto* pm = static_cast<hand::HandKeypoint*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
            else pm->get_postprocessor().set_keypoints_num((int)i);
            break;
        }
        case MD_MODEL_VEHICLE_KEYPOINT: {
            auto* pm = static_cast<landmark::VehicleKeypoint*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
            else pm->get_postprocessor().set_keypoints_num((int)i);
            break;
        }
        case MD_MODEL_OBB: {
            auto* pm = static_cast<detection::UltralyticsObb*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else pm->get_postprocessor().set_nms_threshold((float)d);
            break;
        }
        case MD_MODEL_INSTANCE_SEG: {
            auto* pm = static_cast<detection::UltralyticsSeg*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
            else pm->get_postprocessor().set_mask_threshold((float)d);
            break;
        }
        case MD_MODEL_CLASSIFICATION: {
            auto* pm = static_cast<classification::Classification*>(const_cast<void*>(m));
            if (std::strcmp(name, "top_k") == 0) pm->get_postprocessor().set_top_k((int)i);
            else pm->get_postprocessor().set_multi_label(enable);
            break;
        }
        case MD_MODEL_FACE_DET: {
            auto* pm = static_cast<face::Scrfd*>(const_cast<void*>(m));
            if (std::strcmp(name, "conf_threshold") == 0) pm->get_postprocessor().set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pm->get_postprocessor().set_nms_threshold((float)d);
            else pm->get_postprocessor().set_landmarks_per_face((int)i);
            break;
        }
        case MD_MODEL_OCR_DET: {
            auto* pm = static_cast<ocr::DBDetector*>(const_cast<void*>(m));
            if (std::strcmp(name, "max_side_len") == 0) {
                pm->get_preprocessor().set_max_side_len((int)i);
                break;
            }
            auto& pp = pm->get_postprocessor();
            if (std::strcmp(name, "det_db_thresh") == 0) pp.set_det_db_thresh(d);
            else if (std::strcmp(name, "det_db_box_thresh") == 0) pp.set_det_db_box_thresh(d);
            else if (std::strcmp(name, "det_db_unclip_ratio") == 0) pp.set_det_db_unclip_ratio(d);
            else if (std::strcmp(name, "det_db_score_mode") == 0) pp.set_det_db_score_mode(s);
            else pp.set_use_dilation(enable ? 1 : 0);
            break;
        }
        case MD_MODEL_OCR_CLS: {
            auto* pm = static_cast<ocr::Classifier*>(const_cast<void*>(m));
            pm->get_postprocessor().set_cls_thresh((float)d);
            break;
        }
        case MD_MODEL_OCR: {
            auto* pm = static_cast<ocr::PaddleOCR*>(const_cast<void*>(m));
            if (std::strcmp(name, "max_side_len") == 0) {
                pm->get_detector()->get_preprocessor().set_max_side_len((int)i);
            } else if (std::strcmp(name, "cls_thresh") == 0) {
                pm->get_classifier()->get_postprocessor().set_cls_thresh((float)d);
            } else {
                auto& pp = pm->get_detector()->get_postprocessor();
                if (std::strcmp(name, "det_db_thresh") == 0) pp.set_det_db_thresh(d);
                else if (std::strcmp(name, "det_db_box_thresh") == 0) pp.set_det_db_box_thresh(d);
                else if (std::strcmp(name, "det_db_unclip_ratio") == 0) pp.set_det_db_unclip_ratio(d);
                else if (std::strcmp(name, "det_db_score_mode") == 0) pp.set_det_db_score_mode(s);
                else pp.set_use_dilation(enable ? 1 : 0);
            }
            break;
        }
        case MD_MODEL_PED_ATTR: {
            auto* pm = static_cast<pipeline::PedestrianAttribute*>(const_cast<void*>(m));
            pm->set_det_threshold((float)d);
            break;
        }
        case MD_MODEL_LPR_DET: {
            auto* pm = static_cast<lpr::LprDetection*>(const_cast<void*>(m));
            auto& pp = pm->get_postprocessor();
            if (std::strcmp(name, "conf_threshold") == 0) pp.set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pp.set_nms_threshold((float)d);
            else pp.set_landmarks_per_card((float)d);
            break;
        }
        case MD_MODEL_INSIGHTFACE: {
            auto* pm = static_cast<face::InsightFaceAnalysis*>(const_cast<void*>(m));
            pm->set_det_thresh((float)d);
            break;
        }
        case MD_MODEL_FACE_REC_PIPELINE: {
            auto* pm = static_cast<face::FaceRecognizerPipeline*>(const_cast<void*>(m));
            auto& pp = pm->get_detector()->get_postprocessor();
            if (std::strcmp(name, "conf_threshold") == 0) pp.set_conf_threshold((float)d);
            else if (std::strcmp(name, "nms_threshold") == 0) pp.set_nms_threshold((float)d);
            else pp.set_landmarks_per_face((int)i);
            break;
        }
        default:
            set_error_fmt("md_model_set_param: unsupported kind %d", (int)kind);
            return MD_ERR_UNSUPPORTED_TYPE;
    }
    return MD_OK;
}

} // namespace

MDStatus md_model_set_param_i(MDModelHandle handle, const char* name, int64_t value) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (!name || !*name) { set_error("md_model_set_param_i: name is empty"); return MD_ERR_INVALID_ARGUMENT; }
    return (MDStatus)apply_model_param(mh, name, PT_I, value, 0.0, nullptr);
}

MDStatus md_model_set_param_d(MDModelHandle handle, const char* name, double value) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (!name || !*name) { set_error("md_model_set_param_d: name is empty"); return MD_ERR_INVALID_ARGUMENT; }
    return (MDStatus)apply_model_param(mh, name, PT_D, 0, value, nullptr);
}

MDStatus md_model_set_param_b(MDModelHandle handle, const char* name, int enable) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (!name || !*name) { set_error("md_model_set_param_b: name is empty"); return MD_ERR_INVALID_ARGUMENT; }
    return (MDStatus)apply_model_param(mh, name, PT_B, enable != 0 ? 1 : 0, 0.0, nullptr);
}

MDStatus md_model_set_param_s(MDModelHandle handle, const char* name, const char* value) {
    auto* mh = static_cast<md_model_handle*>(handle);
    if (!mh || !mh->ready) return MD_ERR_MODEL_INIT;
    if (!name || !*name) { set_error("md_model_set_param_s: name is empty"); return MD_ERR_INVALID_ARGUMENT; }
    if (!value) { set_error("md_model_set_param_s: value is null"); return MD_ERR_NULL_POINTER; }
    return (MDStatus)apply_model_param(mh, name, PT_S, 0, 0.0, value);
}

MDStatus md_model_param_names(MDModelKind kind, const char** names) {
    if (!names) return MD_ERR_NULL_POINTER;
    if (kind < 0 || kind >= MD_MODEL_COUNT) {
        set_error("md_model_param_names: invalid kind");
        return MD_ERR_INVALID_ARGUMENT;
    }
    *names = kind_param_names(kind);
    return MD_OK;
}

MDStatus md_model_param_type(MDModelKind kind, const char* name, char* type_out) {
    if (!name || !type_out) return MD_ERR_NULL_POINTER;
    if (kind < 0 || kind >= MD_MODEL_COUNT) {
        set_error("md_model_param_type: invalid kind");
        return MD_ERR_INVALID_ARGUMENT;
    }
    const char t = param_type_of(kind, name);
    if (t == 0) {
        set_error_fmt("md_model_param_type: unknown param '%s' for kind %d", name, (int)kind);
        return MD_ERR_INVALID_ARGUMENT;
    }
    *type_out = t;
    return MD_OK;
}

/* ==================== 结果容器释放 ==================== */

md_result_handle::~md_result_handle() {
    delete static_cast<BatchProjectionBase*>(batch_cache);
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
    *out = origin_count(rh);
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
        case MD_MODEL_HAND: {
            auto* m = static_cast<hand::HandKeypoint*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("hand");
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_VEHICLE_KEYPOINT: {
            auto* m = static_cast<landmark::VehicleKeypoint*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("vehicle_keypoint");
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_LANDMARK: {
            auto* m = static_cast<landmark::FaceLandmark*>(mh->model);
            auto* d = new ResultData<KeyPointsResult>();
            if (!m->predict(image, &d->v)) return predict_fail("face_landmark");
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
        case MD_MODEL_REID: {
            auto* m = static_cast<reid::ReID*>(mh->model);
            auto* d = new ResultData<ReIdResult>();
            std::vector<ReIdResult> r;
            if (!m->predict(image, &r)) return predict_fail("reid");
            for (auto& e : r) d->v.push_back(std::move(e));
            rh->kind = MD_RES_REID;
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
            auto* m = static_cast<face::SeetaFaceAsFirst*>(mh->model);
            auto* d = new ResultData<int>();
            float score = 0.f;
            if (!m->predict(image, &score)) return predict_fail("face anti-spoof first");
            // 单幅整图被动防伪：score > 0.8 判 REAL，否则 SPOOF（与 v1 demo 一致）
            d->v.push_back(score > 0.8f ? static_cast<int>(FaceAntiSpoofResult::REAL)
                                        : static_cast<int>(FaceAntiSpoofResult::SPOOF));
            rh->kind = MD_RES_ANTISPOOF;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AS_SECOND: {
            auto* m = static_cast<face::SeetaFaceAsSecond*>(mh->model);
            auto* d = new ResultData<int>();
            std::vector<std::tuple<int, float>> spoofs;
            if (!m->predict(image, &spoofs)) return predict_fail("face anti-spoof second");
            d->v.push_back(spoofs.empty() ? static_cast<int>(FaceAntiSpoofResult::REAL)
                                          : static_cast<int>(FaceAntiSpoofResult::SPOOF));
            rh->kind = MD_RES_ANTISPOOF;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AS_PIPELINE: {
            auto* m = static_cast<face::SeetaFaceAsPipeline*>(mh->model);
            auto* d = new ResultData<int>();
            std::vector<FaceAntiSpoofResult> labels;
            if (!m->predict(image, &labels)) return predict_fail("face anti-spoof pipeline");
            d->v.reserve(labels.size());
            for (const auto& l : labels) d->v.push_back(static_cast<int>(l));
            rh->kind = MD_RES_ANTISPOOF;
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
        case MD_MODEL_FORMULA_RECOGNIZER: {
            auto* m = static_cast<ocr::FormulaRecognizer*>(mh->model);
            auto* d = new SingleResult<std::string>();
            if (!m->predict(image, &d->value)) return predict_fail("formula recognize");
            rh->kind = MD_RES_FORMULA;
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

MDStatus md_model_predict_batch(MDModelHandle h, MDImageHandle* imgs, size_t n,
                                MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !out) return MD_ERR_NULL_POINTER;
    if (!imgs) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    if (n == 0) {
        set_error("md_model_predict_batch: batch size must be > 0");
        return MD_ERR_INVALID_ARGUMENT;
    }

    auto* rh = new md_result_handle();

    auto predict_fail = [&](const char* what) {
        set_error_fmt("md_model_predict_batch: %s failed", what);
        delete rh;
        return MD_ERR_MODEL_PREDICT;
    };
    auto image_at = [&](size_t i) {
        return handle_to_image(static_cast<md_image_handle*>(imgs[i]));
    };

    switch (mh->kind) {
        case MD_MODEL_DETECTION: {
            auto* m = static_cast<detection::UltralyticsDet*>(mh->model);
            auto* d = new ResultData<std::vector<DetectionResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<DetectionResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("detection");
                d->v.push_back(std::move(r));  // 按图分组
            }
            rh->kind = MD_RES_DETECTION;
            rh->data = d;
            break;
        }
        case MD_MODEL_CLASSIFICATION: {
            auto* m = static_cast<classification::Classification*>(mh->model);
            auto* d = new ResultData<ClassifyResult>();
            for (size_t i = 0; i < n; ++i) {
                ClassifyResult r;
                if (!m->predict(image_at(i), &r)) return predict_fail("classification");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_CLASSIFICATION;
            rh->data = d;
            break;
        }
        case MD_MODEL_POSE: {
            auto* m = static_cast<detection::UltralyticsPose*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("pose");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_HAND: {
            auto* m = static_cast<hand::HandKeypoint*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("hand");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_VEHICLE_KEYPOINT: {
            auto* m = static_cast<landmark::VehicleKeypoint*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("vehicle_keypoint");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_LANDMARK: {
            auto* m = static_cast<landmark::FaceLandmark*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("face_landmark");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_POSE;
            rh->data = d;
            break;
        }
        case MD_MODEL_OBB: {
            auto* m = static_cast<detection::UltralyticsObb*>(mh->model);
            auto* d = new ResultData<std::vector<ObbResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<ObbResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("obb");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_OBB;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSTANCE_SEG: {
            auto* m = static_cast<detection::UltralyticsSeg*>(mh->model);
            auto* d = new ResultData<std::vector<InstanceSegResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<InstanceSegResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("instance seg");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_INSTANCE_SEG;
            rh->data = d;
            break;
        }
        case MD_MODEL_SEM_SEG: {
            auto* m = static_cast<detection::UltralyticsSem*>(mh->model);
            auto* d = new ResultData<SemSegResult>();
            for (size_t i = 0; i < n; ++i) {
                SemSegResult r;
                if (!m->predict(image_at(i), &r)) return predict_fail("sem seg");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_SEM_SEG;
            rh->data = d;
            break;
        }
        case MD_MODEL_DEPTH: {
            auto* m = static_cast<detection::UltralyticsDepth*>(mh->model);
            auto* d = new ResultData<DepthResult>();
            for (size_t i = 0; i < n; ++i) {
                DepthResult r;
                if (!m->predict(image_at(i), &r)) return predict_fail("depth");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_DEPTH;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_DET: {
            auto* m = static_cast<face::Scrfd*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("face det");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_FACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_REC: {
            auto* m = static_cast<face::SeetaFaceID*>(mh->model);
            auto* d = new ResultData<FaceRecognitionResult>();
            for (size_t i = 0; i < n; ++i) {
                FaceRecognitionResult r;
                if (!m->predict(image_at(i), &r)) return predict_fail("face rec");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_FACE_REC;
            rh->data = d;
            break;
        }
        case MD_MODEL_REID: {
            auto* m = static_cast<reid::ReID*>(mh->model);
            auto* d = new ResultData<ReIdResult>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<ReIdResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("reid");
                for (auto& e : r) d->v.push_back(std::move(e));
            }
            rh->kind = MD_RES_REID;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AGE: {
            auto* m = static_cast<face::SeetaFaceAge*>(mh->model);
            auto* d = new ResultData<int>();
            for (size_t i = 0; i < n; ++i) {
                int age = 0;
                if (!m->predict(image_at(i), &age)) return predict_fail("face age");
                d->v.push_back(age);
            }
            rh->kind = MD_RES_AGE;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_GENDER: {
            auto* m = static_cast<face::SeetaFaceGender*>(mh->model);
            auto* d = new ResultData<int>();
            for (size_t i = 0; i < n; ++i) {
                int gender = 0;
                if (!m->predict(image_at(i), &gender)) return predict_fail("face gender");
                d->v.push_back(gender);
            }
            rh->kind = MD_RES_GENDER;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AS: {
            auto* m = static_cast<face::SeetaFaceAsFirst*>(mh->model);
            auto* d = new ResultData<int>();
            for (size_t i = 0; i < n; ++i) {
                float score = 0.f;
                if (!m->predict(image_at(i), &score)) return predict_fail("face anti-spoof first");
                d->v.push_back(score > 0.8f ? static_cast<int>(FaceAntiSpoofResult::REAL)
                                             : static_cast<int>(FaceAntiSpoofResult::SPOOF));
            }
            rh->kind = MD_RES_ANTISPOOF;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_AS_SECOND: {
            set_error_fmt("md_model_predict_batch: batch not implemented for kind %d",
                          (int)mh->kind);
            delete rh;
            return MD_ERR_NOT_IMPLEMENTED;
        }
        case MD_MODEL_FACE_AS_PIPELINE: {
            auto* m = static_cast<face::SeetaFaceAsPipeline*>(mh->model);
            auto* d = new ResultData<int>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<FaceAntiSpoofResult> labels;
                if (!m->predict(image_at(i), &labels)) return predict_fail("face anti-spoof pipeline");
                d->v.reserve(d->v.size() + labels.size());
                for (const auto& l : labels) d->v.push_back(static_cast<int>(l));
            }
            rh->kind = MD_RES_ANTISPOOF;
            rh->data = d;
            break;
        }
        case MD_MODEL_FACE_REC_PIPELINE: {
            auto* m = static_cast<face::FaceRecognizerPipeline*>(mh->model);
            auto* d = new ResultData<std::vector<FaceRecognitionResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<FaceRecognitionResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("face rec pipeline");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_FACE_REC;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSIGHTFACE: {
            auto* m = static_cast<face::InsightFaceAnalysis*>(mh->model);
            auto* d = new ResultData<std::vector<face::InsightFaceResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<face::InsightFaceResult> r;
                if (!m->analyze(image_at(i), &r)) return predict_fail("insightface");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_INSIGHTFACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_INSIGHTFACE_DET: {
            auto* m = static_cast<face::InsightFaceDet*>(mh->model);
            auto* d = new ResultData<std::vector<face::InsightFaceBox>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<face::InsightFaceBox> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("insightface det");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_FACE;
            rh->data = d;
            break;
        }
        case MD_MODEL_OCR:
        case MD_MODEL_OCR_DET:
        case MD_MODEL_OCR_REC:
        case MD_MODEL_OCR_CLS: {
            auto* d = new ResultData<OCRResult>();
            for (size_t i = 0; i < n; ++i) {
                OCRResult r;
                bool ok = false;
                switch (mh->kind) {
                    case MD_MODEL_OCR:
                        ok = static_cast<ocr::PaddleOCR*>(mh->model)->predict(image_at(i), &r);
                        break;
                    case MD_MODEL_OCR_DET:
                        ok = static_cast<ocr::DBDetector*>(mh->model)->predict(image_at(i), &r);
                        break;
                    case MD_MODEL_OCR_REC:
                        ok = static_cast<ocr::Recognizer*>(mh->model)->predict(image_at(i), &r);
                        break;
                    default:
                        ok = static_cast<ocr::Classifier*>(mh->model)->predict(image_at(i), &r);
                        break;
                }
                if (!ok) return predict_fail("ocr");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_OCR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_DET: {
            auto* m = static_cast<lpr::LprDetection*>(mh->model);
            auto* d = new ResultData<std::vector<KeyPointsResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<KeyPointsResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("lpr det");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_REC: {
            auto* m = static_cast<lpr::LprRecognizer*>(mh->model);
            auto* d = new ResultData<LprResult>();
            for (size_t i = 0; i < n; ++i) {
                LprResult r;
                if (!m->predict(image_at(i), &r)) return predict_fail("lpr rec");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_LPR_PIPELINE: {
            auto* m = static_cast<lpr::LprPipeline*>(mh->model);
            auto* d = new ResultData<std::vector<LprResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<LprResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("lpr pipeline");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_LPR;
            rh->data = d;
            break;
        }
        case MD_MODEL_PED_ATTR: {
            auto* m = static_cast<pipeline::PedestrianAttribute*>(mh->model);
            auto* d = new ResultData<std::vector<AttributeResult>>();
            for (size_t i = 0; i < n; ++i) {
                std::vector<AttributeResult> r;
                if (!m->predict(image_at(i), &r)) return predict_fail("ped attr");
                d->v.push_back(std::move(r));
            }
            rh->kind = MD_RES_ATTR;
            rh->data = d;
            break;
        }
        default:
            set_error_fmt("md_model_predict_batch: predict not implemented for kind %d",
                          (int)mh->kind);
            delete rh;
            return MD_ERR_NOT_IMPLEMENTED;
    }

    *out = rh;
    return MD_OK;
}

// 把动作识别模型的类别 scores 装填为 MD_RES_CLASSIFICATION 结果句柄
//（label_ids = 0..N-1，scores = 原始类别得分；复用既有 md_result_classification 读取）。
static MDStatus emit_action_classification(MDModelHandle h, std::vector<float>&& scores,
                                           MDResultHandle* out) {
    auto* d = new ResultData<ClassifyResult>();
    ClassifyResult r;
    r.scores = std::move(scores);
    r.label_ids.reserve(r.scores.size());
    for (size_t i = 0; i < r.scores.size(); ++i)
        r.label_ids.push_back(static_cast<int32_t>(i));
    d->v.push_back(std::move(r));
    auto* rh = new md_result_handle();
    rh->kind = MD_RES_CLASSIFICATION;
    rh->data = d;
    *out = rh;
    return MD_OK;
}

/* ==================== 动作识别（TSN / ST-GCN） ==================== */

MDStatus md_model_predict_sequence(MDModelHandle h, MDImageHandle* frames, size_t n,
                                   MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !out) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    if (mh->kind != MD_MODEL_TSN) {
        set_error("md_model_predict_sequence: only MD_MODEL_TSN supports sequence predict");
        return MD_ERR_UNSUPPORTED_TYPE;
    }
    if (!frames || n == 0) {
        set_error("md_model_predict_sequence: null/empty frames");
        return MD_ERR_INVALID_ARGUMENT;
    }
    auto* m = static_cast<action::TSN*>(mh->model);
    std::vector<ImageData> imgs;
    imgs.reserve(n);
    for (size_t i = 0; i < n; ++i)
        imgs.push_back(handle_to_image(static_cast<md_image_handle*>(frames[i])));
    std::vector<float> scores;
    if (!m->predict(imgs, &scores)) {
        set_error("md_model_predict_sequence: TSN predict failed");
        return MD_ERR_MODEL_PREDICT;
    }
    return emit_action_classification(h, std::move(scores), out);
}

MDStatus md_model_predict_skeleton(MDModelHandle h, const float* joints,
                                   size_t T, size_t V, size_t C, MDResultHandle* out) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !out) return MD_ERR_NULL_POINTER;
    if (!mh->ready) return MD_ERR_MODEL_INIT;
    if (mh->kind != MD_MODEL_ST_GCN) {
        set_error("md_model_predict_skeleton: only MD_MODEL_ST_GCN supports skeleton predict");
        return MD_ERR_UNSUPPORTED_TYPE;
    }
    if (!joints || T == 0 || V == 0 || C == 0 || C > 3) {
        set_error("md_model_predict_skeleton: null joints or invalid T/V/C");
        return MD_ERR_INVALID_ARGUMENT;
    }
    auto* m = static_cast<action::StGcn*>(mh->model);
    // joints 为 T*V*C 行主序（第 t 帧第 v 关节的 C 个坐标）→ KeyPointSeq
    action::KeyPointSeq seq;
    seq.frames.resize(T);
    for (size_t t = 0; t < T; ++t) {
        seq.frames[t].resize(V);
        for (size_t v = 0; v < V; ++v) {
            const float* j = joints + (t * V + v) * C;
            seq.frames[t][v] = modeldeploy::vision::Point3f(j[0], C > 1 ? j[1] : 0.0f, C > 2 ? j[2] : 0.0f);
        }
    }
    std::vector<float> scores;
    if (!m->predict(seq, &scores)) {
        set_error("md_model_predict_skeleton: StGcn predict failed");
        return MD_ERR_MODEL_PREDICT;
    }
    return emit_action_classification(h, std::move(scores), out);
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

MDStatus md_audio_speaker_embed(MDModelHandle h, const float* samples, size_t n,
                                const float** embedding, size_t* emb_n) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !samples || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (n == 0) return MD_ERR_INVALID_ARGUMENT;
    if (!mh->ready || mh->kind != MD_MODEL_SPEAKER_VERIFY) return MD_ERR_INVALID_ARGUMENT;
#ifdef BUILD_AUDIO
    auto* m = static_cast<audio::speaker_verify::SpeakerVerify*>(mh->model);
    // 借用指针生命周期：embedding 存入模型句柄私有 shared_ptr（镜像 reid/result 结果容器持有
    // std::shared_ptr<std::vector<float>> 的所有权方式），由 md_model_destroy 释放，返回指针稳定。
    auto emb = std::make_shared<std::vector<float>>();
    if (!m->predict(std::vector<float>(samples, samples + n), emb.get())) {
        set_error("md_audio_speaker_embed: speaker embed predict failed");
        return MD_ERR_MODEL_PREDICT;
    }
    mh->speaker_embed = std::move(emb);
    *embedding = mh->speaker_embed->data();
    *emb_n = mh->speaker_embed->size();
    return MD_OK;
#else
    (void)samples; (void)n; (void)embedding; (void)emb_n;
    set_error("md_audio_speaker_embed: built without BUILD_AUDIO");
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
    auto* p = project_cached<DetectionResult, MDDetectionItem>(
        rh, [](ProjectedResult<MDDetectionItem>& pp, const std::vector<DetectionResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDDetectionItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score; it.label_id = r.label_id;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_classification(MDResultHandle h, const MDClassifyItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_CLASSIFICATION) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<ClassifyResult, MDClassifyItem>(
        rh, [](ProjectedResult<MDClassifyItem>& pp, const std::vector<ClassifyResult>& srcv) {
            for (const auto& r : srcv) {
                const size_t m = std::min(r.label_ids.size(), r.scores.size());
                for (size_t i = 0; i < m; ++i) {
                    MDClassifyItem it{};
                    it.label_id = r.label_ids[i];
                    it.score = r.scores[i];
                    pp.v.push_back(it);
                }
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_pose(MDResultHandle h, const MDPoseItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<KeyPointsResult, MDPoseItem>(
        rh, [](ProjectedResult<MDPoseItem>& pp, const std::vector<KeyPointsResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDPoseItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_keypoints(MDResultHandle h, size_t i, const MDPoint3** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<KeyPointsResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (kps) *kps = reinterpret_cast<const MDPoint3*>(origin->v[i].keypoints.data());
    if (n) *n = origin->v[i].keypoints.size();
    return MD_OK;
}

MDStatus md_result_obb(MDResultHandle h, const MDObbItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OBB) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<ObbResult, MDObbItem>(
        rh, [](ProjectedResult<MDObbItem>& pp, const std::vector<ObbResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDObbItem it{};
                it.cx = r.rotated_box.xc; it.cy = r.rotated_box.yc;
                it.w = r.rotated_box.width; it.h = r.rotated_box.height;
                it.angle = r.rotated_box.angle;
                it.score = r.score; it.label_id = r.label_id;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_instance_seg(MDResultHandle h, const MDIsegItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<InstanceSegResult, MDIsegItem>(
        rh, [](ProjectedResult<MDIsegItem>& pp, const std::vector<InstanceSegResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDIsegItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score; it.label_id = r.label_id;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_mask(MDResultHandle h, size_t i, const unsigned char** buf, size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !buf) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<InstanceSegResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
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
    // 直接取 handle 持有容器的稳定存储（不再拷贝到局部，否则返回的指针在函数返回后悬垂）
    const SemSegResult* value = nullptr;
    if (auto* s = dynamic_cast<SingleResult<SemSegResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        value = &s->value;
    } else if (auto* d = dynamic_cast<ResultData<SemSegResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        if (!d->v.empty()) value = &d->v[0];  // 批量句柄：读 index 0
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (!value) {
        if (labels) *labels = nullptr;
        if (out_h) *out_h = 0;
        if (out_w) *out_w = 0;
        if (num_classes) *num_classes = 0;
        return MD_OK;
    }
    if (labels) *labels = value->labels.data();
    if (out_h) *out_h = value->shape.empty() ? 0 : static_cast<size_t>(value->shape[0]);
    if (out_w) *out_w = value->shape.size() < 2 ? 0 : static_cast<size_t>(value->shape[1]);
    if (num_classes) *num_classes = value->num_classes;
    return MD_OK;
}

MDStatus md_result_depth(MDResultHandle h, const float** depth, size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !depth) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_DEPTH) return MD_ERR_INVALID_ARGUMENT;
    // 直接取 handle 持有容器的稳定存储（不再拷贝到局部，否则返回的指针在函数返回后悬垂 → AccessViolation）
    const DepthResult* value = nullptr;
    if (auto* s = dynamic_cast<SingleResult<DepthResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        value = &s->value;
    } else if (auto* d = dynamic_cast<ResultData<DepthResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        if (!d->v.empty()) value = &d->v[0];  // 批量句柄：读 index 0
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (!value) {
        if (depth) *depth = nullptr;
        if (out_h) *out_h = 0;
        if (out_w) *out_w = 0;
        return MD_OK;
    }
    if (depth) *depth = value->depth.data();
    if (out_h) *out_h = value->shape.empty() ? 0 : static_cast<size_t>(value->shape[0]);
    if (out_w) *out_w = value->shape.size() < 2 ? 0 : static_cast<size_t>(value->shape[1]);
    return MD_OK;
}

MDStatus md_result_sem_seg_batch(MDResultHandle h, size_t i, const unsigned char** labels, size_t* out_h,
                                 size_t* out_w, int* num_classes) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !labels) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_SEM_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<SemSegResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    const auto& value = d->v[i];
    if (labels) *labels = value.labels.data();
    if (out_h) *out_h = value.shape.empty() ? 0 : static_cast<size_t>(value.shape[0]);
    if (out_w) *out_w = value.shape.size() < 2 ? 0 : static_cast<size_t>(value.shape[1]);
    if (num_classes) *num_classes = value.num_classes;
    return MD_OK;
}

MDStatus md_result_depth_batch(MDResultHandle h, size_t i, const float** depth, size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !depth) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_DEPTH) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<DepthResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    const auto& value = d->v[i];
    if (depth) *depth = value.depth.data();
    if (out_h) *out_h = value.shape.empty() ? 0 : static_cast<size_t>(value.shape[0]);
    if (out_w) *out_w = value.shape.size() < 2 ? 0 : static_cast<size_t>(value.shape[1]);
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
    if (auto* origin = raw_result<KeyPointsResult>(rh)) {
        if (i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].keypoints.data());
        if (n) *n = origin->v[i].keypoints.size();
        return MD_OK;
    }
    if (auto* origin = raw_result<face::InsightFaceBox>(rh)) {
        if (i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].kps.data());
        if (n) *n = origin->v[i].kps.size();
        return MD_OK;
    }
    return MD_ERR_INVALID_ARGUMENT;
}

MDStatus md_result_face_embedding(MDResultHandle h, size_t i,
                                  const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE_REC) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<FaceRecognitionResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (emb_n) *emb_n = origin->v[i].embedding.size();
    if (embedding) *embedding = origin->v[i].embedding.data();
    return MD_OK;
}

MDStatus md_result_reid_embedding(MDResultHandle h, size_t i,
                                  const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_REID) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<ReIdResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (emb_n) *emb_n = origin->v[i].embedding.size();
    if (embedding) *embedding = origin->v[i].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface(MDResultHandle h, const MDInsightFaceItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<face::InsightFaceResult, MDInsightFaceItem>(
        rh, [](ProjectedResult<MDInsightFaceItem>& pp, const std::vector<face::InsightFaceResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDInsightFaceItem it{};
                it.x = r.bbox[0]; it.y = r.bbox[1]; it.w = r.bbox[2] - r.bbox[0]; it.h = r.bbox[3] - r.bbox[1];
                it.score = r.det_score; it.gender = r.gender; it.age = r.age;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_insightface_kps(MDResultHandle h, size_t i, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<face::InsightFaceResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].kps.data());
    if (n) *n = origin->v[i].kps.size();
    return MD_OK;
}

MDStatus md_result_insightface_embedding(MDResultHandle h, size_t i,
                                         const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<face::InsightFaceResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (emb_n) *emb_n = origin->v[i].embedding.size();
    if (embedding) *embedding = origin->v[i].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface_pose(MDResultHandle h, size_t i, const float** pose, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !pose || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<face::InsightFaceResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (n) *n = origin->v[i].pose.size();
    if (pose && !origin->v[i].pose.empty()) *pose = origin->v[i].pose.data();
    return MD_OK;
}

MDStatus md_result_ocr(MDResultHandle h, size_t i, const int** quad, const char** text, float* score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    // 直接取 handle 持有容器的稳定存储（不再拷贝到局部，否则返回的 text/quad 指针在函数返回后悬垂 → 读到空文本/野值）
    const OCRResult* value = nullptr;
    if (auto* s = dynamic_cast<SingleResult<OCRResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        value = &s->value;
    } else if (auto* d = dynamic_cast<ResultData<OCRResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        if (d->v.empty()) return MD_ERR_INVALID_ARGUMENT;
        value = &d->v[0];  // 批量句柄：从第 0 张图读行
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (i >= value->boxes.size()) return MD_ERR_INVALID_ARGUMENT;
    if (quad) *quad = value->boxes[i].data();
    if (text) *text = i < value->text.size() ? value->text[i].c_str() : "";
    if (score) *score = i < value->rec_scores.size() ? value->rec_scores[i] : 0.f;
    return MD_OK;
}

MDStatus md_result_ocr_batch_count(MDResultHandle h, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<OCRResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    *n = d->v.size();  // 图为单位：ResultData<OCRResult> 每图一个
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
    if (auto* origin = raw_result<LprResult>(rh)) {
        if (i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
        if (plate) *plate = origin->v[i].car_plate_str.c_str();
        if (color) *color = origin->v[i].car_plate_color.c_str();
        return MD_OK;
    }
    if (plate) *plate = "";
    if (color) *color = "";
    return MD_OK;
}

MDStatus md_result_ocr_cls(MDResultHandle h, size_t i, int* cls_label, float* cls_score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    int label = 0;
    float score = 0.f;
    if (auto* s = dynamic_cast<SingleResult<OCRResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        const auto& v = s->value;
        label = i < v.cls_labels.size() ? v.cls_labels[i] : 0;
        score = i < v.cls_scores.size() ? v.cls_scores[i] : 0.f;
    } else if (auto* d = dynamic_cast<ResultData<OCRResult>*>(static_cast<ResultDataBase*>(rh->data))) {
        if (!d->v.empty()) {  // 批量句柄：读 index 0
            const auto& v = d->v[0];
            label = i < v.cls_labels.size() ? v.cls_labels[i] : 0;
            score = i < v.cls_scores.size() ? v.cls_scores[i] : 0.f;
        }
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (cls_label) *cls_label = label;
    if (cls_score) *cls_score = score;
    return MD_OK;
}

/* FormulaRecognizer：返回单块 LaTeX（借用指针归结果句柄所有，生命周期同 md_result_plate/ocr 文本） */
MDStatus md_result_formula(MDResultHandle h, size_t i, const char** latex) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !latex) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FORMULA) return MD_ERR_INVALID_ARGUMENT;
    auto* s = dynamic_cast<SingleResult<std::string>*>(static_cast<ResultDataBase*>(rh->data));
    if (!s) return MD_ERR_INVALID_ARGUMENT;
    if (i != 0) return MD_ERR_INVALID_ARGUMENT;
    *latex = s->value.c_str();
    return MD_OK;
}

MDStatus md_result_lpr_keypoints(MDResultHandle h, size_t i, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    if (auto* origin = raw_result<LprResult>(rh)) {
        if (i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
        if (kps) *kps = reinterpret_cast<const MDPoint*>(origin->v[i].keypoints.data());
        if (n) *n = origin->v[i].keypoints.size();
        return MD_OK;
    }
    if (kps) *kps = nullptr;
    if (n) *n = 0;
    return MD_OK;
}

MDStatus md_result_attribute(MDResultHandle h, const MDAttrItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* p = project_cached<AttributeResult, MDAttrItem>(
        rh, [](ProjectedResult<MDAttrItem>& pp, const std::vector<AttributeResult>& srcv) {
            pp.v.reserve(srcv.size());
            for (const auto& r : srcv) {
                MDAttrItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.box_score = r.box_score; it.box_label_id = r.box_label_id;
                pp.v.push_back(it);
            }
        });
    if (!p) return MD_ERR_INVALID_ARGUMENT;
    *items = p->v.data();
    *count = p->v.size();
    return MD_OK;
}

MDStatus md_result_attr_scores(MDResultHandle h, size_t i, const float** scores, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !scores || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* origin = raw_result<AttributeResult>(rh);
    if (!origin || i >= origin->v.size()) return MD_ERR_INVALID_ARGUMENT;
    if (n) *n = origin->v[i].attr_scores.size();
    if (scores) *scores = origin->v[i].attr_scores.data();
    return MD_OK;
}

MDStatus md_result_age(MDResultHandle h, int* age) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_AGE) return MD_ERR_INVALID_ARGUMENT;
    int v = 0;
    if (auto* s = dynamic_cast<SingleResult<int>*>(static_cast<ResultDataBase*>(rh->data))) {
        v = s->value;
    } else if (auto* d = dynamic_cast<ResultData<int>*>(static_cast<ResultDataBase*>(rh->data))) {
        v = d->v.empty() ? 0 : d->v[0];  // 批量句柄：读 index 0
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (age) *age = v;
    return MD_OK;
}

MDStatus md_result_gender(MDResultHandle h, int* gender) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_GENDER) return MD_ERR_INVALID_ARGUMENT;
    int v = 0;
    if (auto* s = dynamic_cast<SingleResult<int>*>(static_cast<ResultDataBase*>(rh->data))) {
        v = s->value;
    } else if (auto* d = dynamic_cast<ResultData<int>*>(static_cast<ResultDataBase*>(rh->data))) {
        v = d->v.empty() ? 0 : d->v[0];  // 批量句柄：读 index 0
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (gender) *gender = v;
    return MD_OK;
}

MDStatus md_result_age_batch(MDResultHandle h, const int** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_AGE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<int>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    *items = d->v.data();
    *count = d->v.size();
    return MD_OK;
}

MDStatus md_result_gender_batch(MDResultHandle h, const int** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_GENDER) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<int>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    *items = d->v.data();
    *count = d->v.size();
    return MD_OK;
}

MDStatus md_result_spoof(MDResultHandle h, size_t i, int* label) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !label) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ANTISPOOF) return MD_ERR_INVALID_ARGUMENT;
    auto* d = raw_result<int>(rh);
    if (!d || i >= d->count()) return MD_ERR_INVALID_ARGUMENT;
    *label = d->v[i];
    return MD_OK;
}

/* ==================== 2D 批量结果 getter（按图索引，逐图取项数组） ==================== */
/* 批量结果来自 md_model_predict_batch。用法：先调各 kind 的 *_batch 数组 getter 取该图项数组，
   再（如需）用 (图,项) 版子项 getter 读 kps/mask/embedding/plate/ocr 行。返回指针在句柄存活期内稳定。 */

MDStatus md_result_detection_batch(MDResultHandle h, size_t img_i, const MDDetectionItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_DETECTION) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<DetectionResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDDetectionItem>(rh, d->v.size(),
        [d](std::vector<MDDetectionItem>& flat, size_t i) {
            for (const auto& r : d->v[i]) {
                MDDetectionItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score; it.label_id = r.label_id;
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_classification_batch(MDResultHandle h, size_t img_i, const MDClassifyItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_CLASSIFICATION) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<ClassifyResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDClassifyItem>(rh, d->v.size(),
        [d](std::vector<MDClassifyItem>& flat, size_t img) {
            const auto& r = d->v[img];
            const size_t m = std::min(r.label_ids.size(), r.scores.size());
            for (size_t k = 0; k < m; ++k) {
                MDClassifyItem it{}; it.label_id = r.label_ids[k]; it.score = r.scores[k];
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_pose_batch(MDResultHandle h, size_t img_i, const MDPoseItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDPoseItem>(rh, d->v.size(),
        [d](std::vector<MDPoseItem>& flat, size_t img) {
            for (const auto& r : d->v[img]) {
                MDPoseItem it{}; it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score; flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_keypoints_batch(MDResultHandle h, size_t img_i, size_t item_j, const MDPoint3** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_POSE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    *kps = reinterpret_cast<const MDPoint3*>(d->v[img_i][item_j].keypoints.data());
    *n = d->v[img_i][item_j].keypoints.size();
    return MD_OK;
}

MDStatus md_result_obb_batch(MDResultHandle h, size_t img_i, const MDObbItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OBB) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<ObbResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDObbItem>(rh, d->v.size(),
        [d](std::vector<MDObbItem>& flat, size_t img) {
            for (const auto& r : d->v[img]) {
                MDObbItem it{};
                it.cx = r.rotated_box.xc; it.cy = r.rotated_box.yc;
                it.w = r.rotated_box.width; it.h = r.rotated_box.height; it.angle = r.rotated_box.angle;
                it.score = r.score; it.label_id = r.label_id;
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_instance_seg_batch(MDResultHandle h, size_t img_i, const MDIsegItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<InstanceSegResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDIsegItem>(rh, d->v.size(),
        [d](std::vector<MDIsegItem>& flat, size_t img) {
            for (const auto& r : d->v[img]) {
                MDIsegItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.score = r.score; it.label_id = r.label_id;
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_mask_batch(MDResultHandle h, size_t img_i, size_t item_j, const unsigned char** buf,
                              size_t* out_h, size_t* out_w) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !buf) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSTANCE_SEG) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<InstanceSegResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    const auto& r = d->v[img_i][item_j];
    *buf = r.mask.buffer.data();
    if (out_h) *out_h = r.mask.shape.empty() ? 0 : static_cast<size_t>(r.mask.shape[0]);
    if (out_w) *out_w = r.mask.shape.size() < 2 ? 0 : static_cast<size_t>(r.mask.shape[1]);
    return MD_OK;
}

MDStatus md_result_face_batch(MDResultHandle h, size_t img_i, const MDFaceItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE) return MD_ERR_INVALID_ARGUMENT;
    BatchProjection<MDFaceItem>* p = nullptr;
    if (auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        p = ensure_batch_proj<MDFaceItem>(rh, d->v.size(),
            [d](std::vector<MDFaceItem>& flat, size_t img) {
                for (const auto& f : d->v[img]) {
                    MDFaceItem it{}; it.x = f.box.x; it.y = f.box.y; it.w = f.box.width; it.h = f.box.height;
                    it.score = f.score; flat.push_back(it);
                }
            });
        if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
        *items = p->flat.data() + p->off[img_i]; *count = p->off[img_i + 1] - p->off[img_i];
        return MD_OK;
    }
    if (auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceBox>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        p = ensure_batch_proj<MDFaceItem>(rh, d->v.size(),
            [d](std::vector<MDFaceItem>& flat, size_t img) {
                for (const auto& f : d->v[img]) {
                    MDFaceItem it{}; it.x = f.bbox[0]; it.y = f.bbox[1]; it.w = f.bbox[2] - f.bbox[0]; it.h = f.bbox[3] - f.bbox[1];
                    it.score = f.score; flat.push_back(it);
                }
            });
        if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
        *items = p->flat.data() + p->off[img_i]; *count = p->off[img_i + 1] - p->off[img_i];
        return MD_OK;
    }
    return MD_ERR_INVALID_ARGUMENT;
}

MDStatus md_result_face_kps_batch(MDResultHandle h, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE) return MD_ERR_INVALID_ARGUMENT;
    if (auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        if (img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
        *kps = reinterpret_cast<const MDPoint*>(d->v[img_i][item_j].keypoints.data());
        *n = d->v[img_i][item_j].keypoints.size();
        return MD_OK;
    }
    if (auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceBox>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        if (img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
        *kps = reinterpret_cast<const MDPoint*>(d->v[img_i][item_j].kps.data());
        *n = d->v[img_i][item_j].kps.size();
        return MD_OK;
    }
    return MD_ERR_INVALID_ARGUMENT;
}

MDStatus md_result_face_embedding_batch(MDResultHandle h, size_t img_i, const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_FACE_REC) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<FaceRecognitionResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    *emb_n = d->v[img_i].embedding.size();
    *embedding = d->v[img_i].embedding.data();
    return MD_OK;
}

MDStatus md_result_reid_embedding_batch(MDResultHandle h, size_t img_i, const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_REID) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<ReIdResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    *emb_n = d->v[img_i].embedding.size();
    *embedding = d->v[img_i].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface_batch(MDResultHandle h, size_t img_i, const MDInsightFaceItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDInsightFaceItem>(rh, d->v.size(),
        [d](std::vector<MDInsightFaceItem>& flat, size_t img) {
            for (const auto& r : d->v[img]) {
                MDInsightFaceItem it{};
                it.x = r.bbox[0]; it.y = r.bbox[1]; it.w = r.bbox[2] - r.bbox[0]; it.h = r.bbox[3] - r.bbox[1];
                it.score = r.det_score; it.gender = r.gender; it.age = r.age;
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_insightface_kps_batch(MDResultHandle h, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    *kps = reinterpret_cast<const MDPoint*>(d->v[img_i][item_j].kps.data());
    *n = d->v[img_i][item_j].kps.size();
    return MD_OK;
}

MDStatus md_result_insightface_embedding_batch(MDResultHandle h, size_t img_i, size_t item_j,
                                               const float** embedding, size_t* emb_n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !embedding || !emb_n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    *emb_n = d->v[img_i][item_j].embedding.size();
    *embedding = d->v[img_i][item_j].embedding.data();
    return MD_OK;
}

MDStatus md_result_insightface_pose_batch(MDResultHandle h, size_t img_i, size_t item_j, const float** pose, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !pose || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_INSIGHTFACE) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<face::InsightFaceResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    *n = d->v[img_i][item_j].pose.size();
    *pose = d->v[img_i][item_j].pose.empty() ? nullptr : d->v[img_i][item_j].pose.data();
    return MD_OK;
}

MDStatus md_result_lpr_batch(MDResultHandle h, size_t img_i, const MDLprItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    BatchProjection<MDLprItem>* p = nullptr;
    if (auto* d = dynamic_cast<ResultData<std::vector<LprResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        p = ensure_batch_proj<MDLprItem>(rh, d->v.size(),
            [d](std::vector<MDLprItem>& flat, size_t img) {
                for (const auto& r : d->v[img]) {
                    MDLprItem it{}; it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                    it.score = r.score; flat.push_back(it);
                }
            });
    } else if (auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        p = ensure_batch_proj<MDLprItem>(rh, d->v.size(),
            [d](std::vector<MDLprItem>& flat, size_t img) {
                for (const auto& r : d->v[img]) {
                    MDLprItem it{}; it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                    it.score = r.score; flat.push_back(it);
                }
            });
    } else {
        return MD_ERR_INVALID_ARGUMENT;
    }
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_plate_batch(MDResultHandle h, size_t img_i, size_t item_j, const char** plate, const char** color) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !plate) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    if (auto* d = dynamic_cast<ResultData<std::vector<LprResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        if (img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
        if (plate) *plate = d->v[img_i][item_j].car_plate_str.c_str();
        if (color) *color = d->v[img_i][item_j].car_plate_color.c_str();
        return MD_OK;
    }
    if (plate) *plate = "";
    if (color) *color = "";
    return MD_OK;
}

MDStatus md_result_lpr_keypoints_batch(MDResultHandle h, size_t img_i, size_t item_j, const MDPoint** kps, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !kps || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_LPR) return MD_ERR_INVALID_ARGUMENT;
    if (auto* d = dynamic_cast<ResultData<std::vector<KeyPointsResult>>*>(static_cast<ResultDataBase*>(rh->data)); d) {
        if (img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
        *kps = reinterpret_cast<const MDPoint*>(d->v[img_i][item_j].keypoints.data());
        *n = d->v[img_i][item_j].keypoints.size();
        return MD_OK;
    }
    return MD_ERR_INVALID_ARGUMENT;
}

MDStatus md_result_attribute_batch(MDResultHandle h, size_t img_i, const MDAttrItem** items, size_t* count) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !items || !count) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<AttributeResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d) return MD_ERR_INVALID_ARGUMENT;
    auto* p = ensure_batch_proj<MDAttrItem>(rh, d->v.size(),
        [d](std::vector<MDAttrItem>& flat, size_t img) {
            for (const auto& r : d->v[img]) {
                MDAttrItem it{};
                it.x = r.box.x; it.y = r.box.y; it.w = r.box.width; it.h = r.box.height;
                it.box_score = r.box_score; it.box_label_id = r.box_label_id;
                flat.push_back(it);
            }
        });
    if (img_i + 1 >= p->off.size()) return MD_ERR_INVALID_ARGUMENT;
    *items = p->flat.data() + p->off[img_i];
    *count = p->off[img_i + 1] - p->off[img_i];
    return MD_OK;
}

MDStatus md_result_attr_scores_batch(MDResultHandle h, size_t img_i, size_t item_j, const float** scores, size_t* n) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh || !scores || !n) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_ATTR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<std::vector<AttributeResult>>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size() || item_j >= d->v[img_i].size()) return MD_ERR_INVALID_ARGUMENT;
    *n = d->v[img_i][item_j].attr_scores.size();
    *scores = d->v[img_i][item_j].attr_scores.data();
    return MD_OK;
}

/* OCR 批量：按 (图, 行) 读第 img_i 张图第 line_j 行（batch 存储为 ResultData<OCRResult>，每图一个） */
MDStatus md_result_ocr_batch(MDResultHandle h, size_t img_i, size_t line_j, const int** quad,
                             const char** text, float* score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<OCRResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    const OCRResult& value = d->v[img_i];
    if (line_j >= value.boxes.size()) return MD_ERR_INVALID_ARGUMENT;
    if (quad) *quad = value.boxes[line_j].data();
    if (text) *text = line_j < value.text.size() ? value.text[line_j].c_str() : "";
    if (score) *score = line_j < value.rec_scores.size() ? value.rec_scores[line_j] : 0.f;
    return MD_OK;
}

MDStatus md_result_ocr_cls_batch(MDResultHandle h, size_t img_i, size_t line_j, int* cls_label, float* cls_score) {
    auto* rh = static_cast<md_result_handle*>(h);
    if (!rh) return MD_ERR_NULL_POINTER;
    if (rh->kind != MD_RES_OCR) return MD_ERR_INVALID_ARGUMENT;
    auto* d = dynamic_cast<ResultData<OCRResult>*>(static_cast<ResultDataBase*>(rh->data));
    if (!d || img_i >= d->v.size()) return MD_ERR_INVALID_ARGUMENT;
    const OCRResult& value = d->v[img_i];
    if (cls_label) *cls_label = line_j < value.cls_labels.size() ? value.cls_labels[line_j] : 0;
    if (cls_score) *cls_score = line_j < value.cls_scores.size() ? value.cls_scores[line_j] : 0.f;
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
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_draw_rect: device frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
    const cv::Scalar cv_color = md_color_to_scalar(color);
    // 直接复用 C++ 的绘制实现（与 vis_* 系一致的 alpha 混合）；mat 为借用视图，写回句柄底层缓冲
    modeldeploy::vision::draw_filled_rect(mat, {cvRound(x), cvRound(y), cvRound(w), cvRound(h)},
                                          cv_color, alpha);
    return MD_OK;
}

MDStatus md_draw_polygon(MDImageHandle img, const float* xs, const float* ys, size_t n,
                         MDColorRGBA color, float alpha) {
    auto* hi = static_cast<md_image_handle*>(img);
    if (!hi || !xs || !ys) return MD_ERR_NULL_POINTER;
    if (n < 3) return MD_ERR_INVALID_ARGUMENT;
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_draw_polygon: device frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
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
    cv::Mat mat;
    if (!hi->image.asMat(&mat)) { set_error("md_draw_text: device frame not supported"); return MD_ERR_UNSUPPORTED_TYPE; }
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

// 就地绘制：返回 ImageHandle 的统一 ImageData（浅拷贝共享底层），vis_* 会写回
ImageData image_handle_as_data(md_image_handle* hi) {
    return hi->image;
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

    // 设备帧（NV12 且非 CPU）：就地设备绘制，按 frame.device() 分发到对应 processor backend
    // （GPU→CUDA kernel，TPU→Sophgo，其余→CPU 顺序实现）。
    if (image.device() != Device::CPU && image.type() == MdImageType::NV12) {
        const double threshold = opt.threshold > 0 ? opt.threshold : 0.5;
        auto draw_backend = modeldeploy::vision::create_processor_backend(
            image.device(), modeldeploy::Backend::SOPHGO, 0);
        // 校验该设备是否真返回了对应设备后端（而非回退到 CPU 顺序后端）。注意：
        // Cuda/SophgoProcessorBackend 均继承自 CpuProcessorBackend，故不能用
        // dynamic_cast<CpuProcessorBackend*> 判断回退（那会误杀设备绘制）；
        // 必须按 device 精确匹配期望的具体后端。若设备未启用对应后端
        // （如 WITH_GPU/OFF、非 TPU），工厂回退到 CPU，但 y()/uv() 指向设备内存，
        // CPU 后端用宿主指针写入会越界/UB —— 直接拒绝。
        bool device_backend_ready = false;
#ifdef WITH_GPU
        if (image.device() == Device::GPU) {
            device_backend_ready =
                dynamic_cast<modeldeploy::vision::CudaProcessorBackend*>(draw_backend.get()) != nullptr;
        }
#endif
#ifdef ENABLE_SOPHGO
        if (image.device() == Device::TPU) {
            device_backend_ready =
                dynamic_cast<modeldeploy::vision::SophgoProcessorBackend*>(draw_backend.get()) != nullptr;
        }
#endif
        if (!device_backend_ready) {
            set_error_fmt("md_draw_result: backend for device %d unavailable (fallback to CPU cannot draw device memory)",
                          (int)image.device());
            return MD_ERR_NOT_IMPLEMENTED;
        }
        bool ok = false;
        switch (rh->kind) {
            case MD_RES_DETECTION: {
                auto* d = raw_result<DetectionResult>(rh);
                if (!d) return MD_ERR_INVALID_ARGUMENT;
                for (const auto& r : d->v) {
                    if (r.score < threshold) continue;
                    const auto& box = r.box;
                    ok = draw_backend->draw_rect_nv12(image, box.x, box.y, box.width, box.height,
                                                      255, 0, 0, 2) || ok;
                    ok = draw_backend->draw_text_nv12(image, box.x, box.y - 16,
                                                      std::to_string(r.label_id), 255, 255, 255, 1) || ok;
                }
                return ok ? MD_OK : MD_ERR_INVALID_ARGUMENT;
            }
            default:
                set_error_fmt("md_draw_result: device draw for kind %d not yet implemented", (int)rh->kind);
                return MD_ERR_NOT_IMPLEMENTED;
        }
    }

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

/* ==================== 跟踪器 ==================== */

MDStatus md_tracker_create(MDTrackerKind kind, MDTrackerHandle* out) {
    if (!out) { set_error("md_tracker_create: out is null"); return MD_ERR_NULL_POINTER; }
    using namespace modeldeploy::vision::tracking;
    auto* h = new md_tracker_handle();
    h->kind = kind;
    switch (kind) {
        case MD_TRACKER_BYTETRACK: h->tracker = std::make_unique<ByteTracker>(); break;
        case MD_TRACKER_BOTSORT:   h->tracker = std::make_unique<BotSortTracker>(); break;
        case MD_TRACKER_STRONGSORT: h->tracker = std::make_unique<StrongSortTracker>(); break;
        default:
            delete h;
            set_error_fmt("md_tracker_create: unsupported kind %d", (int)kind);
            return MD_ERR_INVALID_ARGUMENT;
    }
    apply_tracker_params(h->tracker.get(), kind, h->params);
    *out = h;
    return MD_OK;
}

void md_tracker_destroy(MDTrackerHandle h) {
    delete static_cast<md_tracker_handle*>(h);
}

namespace {
// Converts capi box array into the tracking Detection vector (shared by the
// non-mutating capacity query and the stateful update commit).
std::vector<modeldeploy::vision::tracking::Detection> tracker_build_dets(
    const MDBox* boxes, const float* scores, const int* label_ids, size_t n) {
    using namespace modeldeploy::vision::tracking;
    std::vector<Detection> dets;
    dets.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Detection d;
        d.box = Rect2f(boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
        d.score = scores ? scores[i] : 0.f;
        d.label_id = label_ids ? label_ids[i] : 0;
        dets.push_back(std::move(d));
    }
    return dets;
}
} // namespace

/* 非变异容量查询：计算对给定检测调用 update(n) 会产生的 TrackResult 数量，
 * 但**不推进**跟踪器状态（帧计数 / Kalman / 关联）。实现为对跟踪器做一次克隆后
 * 在克隆上运行 update，从而得到与真实 update 完全一致的输出数。
 * 调用方应先调用本函数确定缓冲大小，再分配并用该容量调用 stateful md_tracker_update
 * —— 这保证 update 每逻辑帧只提交一次（避免两阶段容量探测导致的双重推进）。 */
MDStatus md_tracker_capacity(MDTrackerHandle h, const MDBox* boxes, const float* scores,
                             const int* label_ids, size_t n, size_t* out_count) {
    auto* th = static_cast<md_tracker_handle*>(h);
    if (!th || !th->tracker) { set_error("md_tracker_capacity: handle is null"); return MD_ERR_NULL_POINTER; }
    if (!out_count) { set_error("md_tracker_capacity: out_count is null"); return MD_ERR_NULL_POINTER; }
    if (n > 0 && (!boxes || !scores || !label_ids)) {
        set_error("md_tracker_capacity: boxes/scores/label_ids null for non-empty n");
        return MD_ERR_NULL_POINTER;
    }
    auto dets = tracker_build_dets(boxes, scores, label_ids, n);
    try {
        std::unique_ptr<modeldeploy::vision::tracking::BaseTracker> probe = th->tracker->clone();
        std::vector<modeldeploy::vision::tracking::TrackResult> res = probe->update(dets);
        *out_count = res.size();
    } catch (const std::exception& e) {
        set_error_fmt("md_tracker_capacity: %s", e.what());
        return MD_ERR_MODEL_PREDICT;
    }
    return MD_OK;
}

/* 有状态更新提交：推进跟踪器一帧并写入输出。
 * 调用方应先经 md_tracker_capacity 得到所需数量并分配足够缓冲，再调用本函数（每逻辑帧
 * 恰好一次）。容量不足时本函数会在**克隆**上预估所需数、写 *out_count 为需要数并返回
 * MD_ERR_INVALID_ARGUMENT，且**不推进**真实跟踪器状态（帧计数 / Kalman / 关联），
 * 从而保证容量探测永不导致双重推进。 */
MDStatus md_tracker_update(MDTrackerHandle h, const MDBox* boxes, const float* scores,
                           const int* label_ids, size_t n, MDTrackItem* out, size_t* out_count) {
    auto* th = static_cast<md_tracker_handle*>(h);
    if (!th || !th->tracker) { set_error("md_tracker_update: handle is null"); return MD_ERR_NULL_POINTER; }
    if (!out_count) { set_error("md_tracker_update: out_count is null"); return MD_ERR_NULL_POINTER; }
    if (!out) { set_error("md_tracker_update: out is null"); return MD_ERR_NULL_POINTER; }
    if (n > 0 && (!boxes || !scores || !label_ids)) {
        set_error("md_tracker_update: boxes/scores/label_ids null for non-empty n");
        return MD_ERR_NULL_POINTER;
    }

    using namespace modeldeploy::vision::tracking;
    auto dets = tracker_build_dets(boxes, scores, label_ids, n);
    const size_t cap = *out_count;

    try {
        // 先做**非变异**容量预估：在克隆上跑一次 update 得所需数量，不触碰真实状态。
        // 仅当容量充足时才提交真实的有状态 update；容量不足则直接返回，保证本函数
        // 不会在容量未命中时推进帧状态（这正是查询/提交契约的核心保证）。
        std::unique_ptr<BaseTracker> probe = th->tracker->clone();
        const size_t need = probe->update(dets).size();
        if (cap < need) {
            *out_count = need;
            set_error_fmt("md_tracker_update: output capacity %zu < needed %zu (track count)", cap, need);
            return MD_ERR_INVALID_ARGUMENT;  // 未推进真实跟踪器状态
        }

        std::vector<TrackResult> res = th->tracker->update(dets);
        const size_t written = res.size();
        for (size_t i = 0; i < written; ++i) {
            MDTrackItem it;
            it.x = res[i].box.x; it.y = res[i].box.y;
            it.w = res[i].box.width; it.h = res[i].box.height;
            it.track_id = res[i].track_id;
            it.label_id = res[i].label_id;
            it.score = res[i].score;
            it.state = res[i].state;
            out[i] = it;
        }
        *out_count = written;
    } catch (const std::exception& e) {
        set_error_fmt("md_tracker_update: %s", e.what());
        return MD_ERR_MODEL_PREDICT;
    }
    return MD_OK;
}

MDStatus md_tracker_set_params(MDTrackerHandle h, const char* name, double value) {
    auto* th = static_cast<md_tracker_handle*>(h);
    if (!th || !th->tracker) { set_error("md_tracker_set_params: handle is null"); return MD_ERR_NULL_POINTER; }
    if (!name || !*name) { set_error("md_tracker_set_params: name is empty"); return MD_ERR_INVALID_ARGUMENT; }

    TrackerParams& p = th->params;
    std::string n(name);
    if (n == "track_thresh") p.track_thresh = (float)value;
    else if (n == "high_thresh") p.high_thresh = (float)value;
    else if (n == "low_thresh") p.low_thresh = (float)value;
    else if (n == "max_age") p.max_age = (int)value;
    else if (n == "min_hits") p.min_hits = (int)value;
    else if (n == "iou_threshold") p.iou_threshold = (float)value;
    else if (n == "match_thresh") p.match_thresh = (float)value;
    else if (n == "ema_alpha") p.ema_alpha = (float)value;
    else if (n == "fuse_score_weight") p.fuse_score_weight = (float)value;
    else if (n == "appearance_priority") p.appearance_priority = (float)value;
    else if (n == "with_cmc") p.with_cmc = (value != 0.0);
    else {
        set_error_fmt("md_tracker_set_params: unsupported parameter '%s'", name);
        return MD_ERR_INVALID_ARGUMENT;
    }
    apply_tracker_params(th->tracker.get(), th->kind, p);
    return MD_OK;
}

MDStatus md_tracker_reset(MDTrackerHandle h) {
    auto* th = static_cast<md_tracker_handle*>(h);
    if (!th || !th->tracker) { set_error("md_tracker_reset: handle is null"); return MD_ERR_NULL_POINTER; }
    th->tracker->reset();
    return MD_OK;
}

/* ==================== 条码 / 二维码识别 ==================== */

MDStatus md_barcode_create(MDBarcodeHandle* out) {
    if (!out) { set_error("md_barcode_create: out is null"); return MD_ERR_NULL_POINTER; }
    auto* h = new md_barcode_handle();
    h->formats = modeldeploy::vision::barcode::FMT_ALL;
    h->det.set_formats(h->formats);
    *out = h;
    return MD_OK;
}

void md_barcode_destroy(MDBarcodeHandle h) {
    delete static_cast<md_barcode_handle*>(h);
}

MDStatus md_barcode_set_formats(MDBarcodeHandle h, uint32_t formats) {
    if (!h) { set_error("md_barcode_set_formats: h is null"); return MD_ERR_NULL_POINTER; }
    auto* bh = static_cast<md_barcode_handle*>(h);
    bh->formats = formats;
    bh->det.set_formats(formats);
    return MD_OK;
}

/* 无状态 detect：不推进任何状态，故容量查询与写入可安全用同一调用（纯函数）。
 * 容量查询：items==nullptr 时仅置 *count 为需要数，返回 MD_OK。 */
MDStatus md_barcode_detect(MDBarcodeHandle h, MDImageHandle img,
                           MD_BarcodeItem* items, uint32_t* count) {
    if (!h || !img || !count) {
        set_error("md_barcode_detect: null argument");
        return MD_ERR_NULL_POINTER;
    }
    const modeldeploy::vision::ImageData image =
        handle_to_image(static_cast<md_image_handle*>(img));
    auto res = static_cast<md_barcode_handle*>(h)->det.detect(image);
    const uint32_t need = static_cast<uint32_t>(res.size());
    if (items == nullptr) { *count = need; return MD_OK; }  // 容量查询
    const uint32_t cap = *count;
    const uint32_t n = std::min(cap, need);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& r = res[i];
        MD_BarcodeItem& it = items[i];
        memset(&it, 0, sizeof(it));
        memcpy(it.text, r.text.c_str(), std::min<size_t>(r.text.size(), sizeof(it.text) - 1));
        memcpy(it.format, r.format.c_str(), std::min<size_t>(r.format.size(), sizeof(it.format) - 1));
        for (int k = 0; k < 4; ++k) {
            it.quad[2 * k] = r.quad[k].x;
            it.quad[2 * k + 1] = r.quad[k].y;
        }
        it.score = r.score;
        it.is_qr = r.is_qr ? 1 : 0;
    }
    *count = n;
    return MD_OK;
}

/* ==================== 解决方案（vision::solution / tool） ==================== */

struct md_solution_handle {
    MDSolutionKind kind;
    void* obj;
};

MDStatus md_solution_create(MDSolutionHandle* out, MDSolutionKind kind) {
    if (!out) return MD_ERR_NULL_POINTER;
    auto* h = new md_solution_handle();
    h->kind = kind;
#ifdef BUILD_VISION
    switch (kind) {
      case MD_SOLUTION_OBJECT_COUNTER: h->obj = new vision::solution::ObjectCounter(); break;
      case MD_SOLUTION_HEATMAP:        h->obj = new vision::solution::Heatmap(); break;
      case MD_SOLUTION_SPEED:          h->obj = new vision::solution::SpeedEstimator(); break;
      case MD_SOLUTION_DISTANCE:       h->obj = new vision::solution::DistanceEstimator(); break;
      case MD_SOLUTION_WORKOUT:        h->obj = new vision::solution::WorkoutMonitor(); break;
      case MD_SOLUTION_PARKING:        h->obj = new vision::solution::ParkingManager(); break;
      default: delete h; return MD_ERR_INVALID_ARGUMENT;
    }
    *out = h;
    return MD_OK;
#else
    delete h; set_error("built without BUILD_VISION"); return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_destroy(MDSolutionHandle h) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    switch (h->kind) {
      case MD_SOLUTION_OBJECT_COUNTER: delete static_cast<vision::solution::ObjectCounter*>(h->obj); break;
      case MD_SOLUTION_HEATMAP:        delete static_cast<vision::solution::Heatmap*>(h->obj); break;
      case MD_SOLUTION_SPEED:          delete static_cast<vision::solution::SpeedEstimator*>(h->obj); break;
      case MD_SOLUTION_DISTANCE:       delete static_cast<vision::solution::DistanceEstimator*>(h->obj); break;
      case MD_SOLUTION_WORKOUT:        delete static_cast<vision::solution::WorkoutMonitor*>(h->obj); break;
      case MD_SOLUTION_PARKING:        delete static_cast<vision::solution::ParkingManager*>(h->obj); break;
      default: break;
    }
#endif
    delete h;
    return MD_OK;
}

MDStatus md_solution_object_counter_set_line(MDSolutionHandle h, float ax, float ay, float bx, float by) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    static_cast<vision::solution::ObjectCounter*>(h->obj)
        ->set_line(vision::Point2f(ax, ay), vision::Point2f(bx, by));
    return MD_OK;
#else
    (void)ax;(void)ay;(void)bx;(void)by; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_object_counter_update(MDSolutionHandle h, const float* boxes, size_t n,
                                           const int* label_ids, const int* track_ids) {
    if (!h || !boxes || n == 0 || !label_ids || !track_ids) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    auto* c = static_cast<vision::solution::ObjectCounter*>(h->obj);
    std::vector<tracking::TrackResult> tracks(n);
    for (size_t i = 0; i < n; ++i) {
        tracks[i].track_id = track_ids[i];
        tracks[i].box = vision::Rect2f(boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]);
        tracks[i].label_id = label_ids[i];
        tracks[i].score = 1.0f;
    }
    c->update(tracks);
    return MD_OK;
#else
    (void)boxes;(void)n;(void)label_ids;(void)track_ids; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_object_counter_hline(MDSolutionHandle h, int* in, int* out_count) {
    if (!h || !in || !out_count) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    auto st = static_cast<vision::solution::ObjectCounter*>(h->obj)->stats();
    *in = st.line_in; *out_count = st.line_out;
    return MD_OK;
#else
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_set_size(MDSolutionHandle h, int w, int hh) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    static_cast<vision::solution::Heatmap*>(h->obj)->set_size(w, hh);
    return MD_OK;
#else
    (void)w;(void)hh; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_update(MDSolutionHandle h, const float* boxes, size_t n, int frame_w, int frame_h) {
    if (!h || !boxes || n == 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    auto* hm = static_cast<vision::solution::Heatmap*>(h->obj);
    std::vector<tracking::TrackResult> tracks(n);
    for (size_t i = 0; i < n; ++i) {
        tracks[i].track_id = (int)i;
        tracks[i].box = vision::Rect2f(boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]);
    }
    hm->update(tracks, frame_w, frame_h);
    return MD_OK;
#else
    (void)boxes;(void)n;(void)frame_w;(void)frame_h; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_peak(MDSolutionHandle h, int* x, int* y) {
    if (!h || !x || !y) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    auto p = static_cast<vision::solution::Heatmap*>(h->obj)->peak();
    *x = p.first; *y = p.second;
    return MD_OK;
#else
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_vision_iou4(float ax, float ay, float aw, float ah,
                        float bx, float by, float bw, float bh, float* out) {
    if (!out) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    *out = vision::tool::iou(vision::Rect2f(ax,ay,aw,ah), vision::Rect2f(bx,by,bw,bh));
    return MD_OK;
#else
    (void)ax;(void)ay;(void)aw;(void)ah;(void)bx;(void)by;(void)bw;(void)bh;
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

/* ==================== 音频解决方案（audio::solution / tool） ==================== */

static std::vector<float> g_resample_buf;

struct md_audio_solution_handle { MDAudioSolutionKind kind; void* obj; };

MDStatus md_audio_solution_create(MDAudioSolutionHandle* out, MDAudioSolutionKind kind) {
    if (!out) return MD_ERR_NULL_POINTER;
    auto* h = new md_audio_solution_handle(); h->kind = kind;
#ifdef BUILD_AUDIO
    switch (kind) {
      case MD_AUDIO_SPEAKER_SEARCH: h->obj = new audio::solution::SpeakerSearch(); break;
      case MD_AUDIO_TTS_BATCHER:    h->obj = new audio::solution::TTSBatcher(); break;
      default: delete h; return MD_ERR_INVALID_ARGUMENT;
    }
    *out = h; return MD_OK;
#else
    delete h; set_error("built without BUILD_AUDIO"); return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_solution_destroy(MDAudioSolutionHandle h) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    switch (h->kind) {
      case MD_AUDIO_SPEAKER_SEARCH: delete static_cast<audio::solution::SpeakerSearch*>(h->obj); break;
      case MD_AUDIO_TTS_BATCHER:    delete static_cast<audio::solution::TTSBatcher*>(h->obj); break;
      default: break;
    }
#endif
    delete h; return MD_OK;
}

MDStatus md_audio_speaker_search_enroll(MDAudioSolutionHandle h, const char* label,
                                        const float* emb, size_t n) {
    if (!h || !label || !emb || n == 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    if (h->kind != MD_AUDIO_SPEAKER_SEARCH) return MD_ERR_INVALID_ARGUMENT;
    static_cast<audio::solution::SpeakerSearch*>(h->obj)->enroll(label, std::vector<float>(emb, emb + n));
    return MD_OK;
#else
    (void)label;(void)emb;(void)n; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_speaker_search_match(MDAudioSolutionHandle h, const float* emb, size_t n, int k,
                                       const char** best_label, float* best_score) {
    if (!h || !emb || n == 0 || !best_label || !best_score) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    if (h->kind != MD_AUDIO_SPEAKER_SEARCH) return MD_ERR_INVALID_ARGUMENT;
    auto r = static_cast<audio::solution::SpeakerSearch*>(h->obj)->match(std::vector<float>(emb, emb + n), k);
    if (r.empty()) return MD_ERR_MODEL_PREDICT;
    static std::string g_label;
    g_label = r[0].first;
    *best_label = g_label.c_str();
    *best_score = r[0].second;
    return MD_OK;
#else
    (void)emb;(void)n;(void)k;(void)best_label;(void)best_score; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_resample(const float* in, size_t n, int in_sr, int out_sr,
                           float** out, size_t* out_n) {
    if (!in || !out || !out_n || n == 0 || in_sr <= 0 || out_sr <= 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    g_resample_buf = audio::tool::Resampler::resample(std::vector<float>(in, in + n), in_sr, out_sr);
    *out = g_resample_buf.data(); *out_n = g_resample_buf.size();
    return MD_OK;
#else
    (void)in_sr;(void)out_sr; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_meta(const char* wav, int* sample_rate, int* channels, int* bits, uint32_t* duration_ms) {
    if (!wav || !sample_rate || !channels || !bits || !duration_ms) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    auto meta = audio::tool::parse_meta(wav);
    *sample_rate = meta.sample_rate; *channels = meta.channels;
    *bits = meta.bits; *duration_ms = meta.duration_ms;
    return MD_OK;
#else
    (void)wav; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}
