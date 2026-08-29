//
// Created by aichao on 2025/8/2.
//

#include "core/md_log.h"
#include "vision/processors/cuda/cuda_processor_backend.h"
#include "vision/processors/cuda/yolo_preproc.cuh"
#include "vision/processors/cuda/fused_preproc.cuh"
#include "vision/processors/cuda/scrfd_preproc.cuh"
#include "vision/processors/cuda/draw_gpu.cuh"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
// 与 CPU vis_* 确定性对齐的类调色板(RGB 序;20 色,label_id%20 取)
constexpr uint8_t kClassPalette[20][3] = {
    {230, 159, 0}, {86, 180, 233}, {0, 158, 115}, {240, 228, 66},
    {0, 114, 178}, {213, 94, 0}, {204, 121, 167}, {255, 157, 0},
    {35, 105, 153}, {166, 86, 40}, {247, 129, 191}, {120, 87, 156},
    {255, 154, 161}, {107, 174, 214}, {222, 125, 44}, {152, 78, 163},
    {191, 119, 0}, {32, 74, 135}, {204, 154, 72}, {188, 143, 143}
};
// label_id 可能为负（无效类别）→ C++ % 结果也为负，`label_id % 20` 会前越界读数组；钳到 [0,20)
static inline int palette_idx(int label_id) { return label_id >= 0 ? label_id % 20 : 0; }

std::string label_name(const modeldeploy::vision::VisionProcessorBackend::VisOptions& opt, int label_id) {
    auto it = opt.label_map.find(label_id);
    if (it != opt.label_map.end() && !it->second.empty()) return it->second;
    return std::to_string(label_id);
}
std::string score_str(float score) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.2f", score);
    return std::string(buf);
}

// ── 标签绘制：优先用 OpenCV(cv::FontFace/真 TTF/UTF-8/抗锯齿/字号颜色)，无 font_path 时回退 8x16 位图 ──
constexpr int   kLabelFontPx   = 30;      // OpenCV 文本像素高度
constexpr float kLabelBgAlpha  = 0.75f;
static inline bool label_is_ascii(const std::string& s) {
    for (unsigned char ch : s) if (ch < 0x20 || ch > 0x7E) return false;
    return true;
}
static inline int label_char_w(bool ascii, int fs) { return (ascii ? 8 : 16) * fs; }
static inline int label_bitmap_fs(int font_px) { return std::max(1, font_px / 16); }

// 用 OpenCV(cv::FontFace)把标签(不透明压暗底块 + 白字,抗锯齿)渲染成紧贴墨迹的 BGR sprite。
// 返回独立自持(已 clone)的 sprite；失败/字体缺失返回空 Mat。
static cv::Mat render_text_sprite(const std::string& text, const uint8_t* bgr_tag, int font_px,
                                  const std::string& font_path) {
    cv::Mat empty;
    if (text.empty() || font_path.empty()) return empty;
    static std::unordered_map<std::string, cv::FontFace> font_cache;
    auto it = font_cache.find(font_path);
    if (it == font_cache.end())
        it = font_cache.emplace(font_path, cv::FontFace(font_path)).first;
    const int cw = static_cast<int>(text.size()) * (font_px + 6) + 64;
    const int ch = 2 * font_px + 32;
    // 压暗底色保证白字在亮色(黄等)上也可读；bgr_tag 按设备约定为 (R,G,B)，cv::Scalar 为 (B,G,R)
    static constexpr double kTagDark = 0.60;
    const uint8_t bg_b = static_cast<uint8_t>(std::lround(bgr_tag[2] * kTagDark));
    const uint8_t bg_g = static_cast<uint8_t>(std::lround(bgr_tag[1] * kTagDark));
    const uint8_t bg_r = static_cast<uint8_t>(std::lround(bgr_tag[0] * kTagDark));
    cv::Mat canvas(ch, cw, CV_8UC3, cv::Scalar(bg_b, bg_g, bg_r));
    cv::putText(canvas, text, cv::Point(8, font_px + 14), cv::Scalar(255, 255, 255),
                it->second, font_px, 600);
    // 墨迹掩码：任一通道 != 底色即视为文本(含 AA 边缘；canvas 底色为压暗后 BGR: bg_b,bg_g,bg_r)
    cv::Mat mask = cv::Mat::zeros(ch, cw, CV_8UC1);
    for (int i = 0; i < cw * ch; ++i) {
        const uint8_t* p = canvas.data + i * 3;
        mask.data[i] = (p[0] != bg_b || p[1] != bg_g || p[2] != bg_r) ? 255 : 0;
    }
    cv::Rect bbox = cv::boundingRect(mask);
    if (bbox.width <= 0 || bbox.height <= 0) return empty;
    const int pad = 3;
    int x0 = std::max(0, bbox.x - pad), y0 = std::max(0, bbox.y - pad);
    int x1 = std::min(cw, bbox.x + bbox.width + pad), y1 = std::min(ch, bbox.y + bbox.height + pad);
    return canvas(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
}

// 把 host sprite blit 到 NV12 指定 top-left（只夹取），并确保 src 步长正确。
static bool blit_sprite(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                        cudaStream_t s, const cv::Mat& sprite, int ox, int oy) {
    if (sprite.empty()) return false;
    if (ox + sprite.cols > w) ox = std::max(0, w - sprite.cols);
    if (oy + sprite.rows > h) oy = std::max(0, h - sprite.rows);
    if (ox < 0) ox = 0; if (oy < 0) oy = 0;
    return modeldeploy::vision::blit_bgr_nv12_gpu(y, uv, w, h, step_y, step_uv,
                                                  sprite.data, sprite.cols, sprite.rows,
                                                  static_cast<int>(sprite.step), ox, oy, s);
}

// 渲染 label sprite，并把其底边压到检测框顶边框上(重叠 overlap px，消除视觉空隙)；上方放不下则折入框内顶部。
static bool blit_sprite_above_box(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                                  cudaStream_t s, const std::string& text, const uint8_t* rgb,
                                  int font_px, const std::string& font_path,
                                  float box_x, float box_y, int overlap) {
    const cv::Mat sp = render_text_sprite(text, rgb, font_px, font_path);
    if (sp.empty()) return false;
    // 底边 = 框顶 + overlap(压住边框厚度)，故无可见间隙
    int by = static_cast<int>(box_y) - sp.rows + overlap;
    if (by < 0) by = static_cast<int>(box_y) + 2;
    return blit_sprite(y, uv, w, h, step_y, step_uv, s, sp, static_cast<int>(box_x), by);
}

// 绘制一个带底块的标签条；x,y 为底块 top-left（只夹取，不自动移动）。
static bool draw_label_tag(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                           cudaStream_t s, int x, int y_pos, const std::string& text,
                           const uint8_t* rgb, int font_px, const std::string& font_path) {
    if (text.empty()) return true;
    const cv::Mat sp = render_text_sprite(text, rgb, font_px, font_path);
    if (!sp.empty())
        return blit_sprite(y, uv, w, h, step_y, step_uv, s, sp, x, y_pos);
    // 位图回退
    const bool ascii = label_is_ascii(text);
    const int fs = label_bitmap_fs(font_px);
    const int charw = label_char_w(ascii, fs);
    const int th = 16 * fs;
    const int tw = static_cast<int>(text.size()) * charw;
    int bx = x, by = y_pos;
    if (bx < 2) bx = 2;
    if (bx + tw + 4 > w - 2) bx = std::max(2, w - 4 - tw);
    if (by < 0) by = 0;
    if (by + th + 2 > h - 2) by = std::max(0, h - th - 2);
    bool ok = modeldeploy::vision::fill_rect_nv12_gpu(y, uv, w, h, step_y, step_uv,
                                 bx - 2, by - 2, bx + tw + 2, by + th,
                                 rgb ? rgb[0] : 0, rgb ? rgb[1] : 0, rgb ? rgb[2] : 0,
                                 kLabelBgAlpha, s);
    if (ascii)
        ok = modeldeploy::vision::draw_text_nv12_gpu(y, uv, w, h, step_y, step_uv, (float)bx, (float)by,
                                text.c_str(), 255, 255, 255, fs, s) && ok;
    else
        ok = modeldeploy::vision::draw_text_cjk_nv12_gpu(y, uv, w, h, step_y, step_uv, (float)bx, (float)by,
                                    text.c_str(), 255, 255, 255, fs,
                                    static_cast<int>(text.size()) + 4, s) && ok;
    return ok;
}

// 在检测框左上角排放标签：优先框上方，底边与框顶留 2px；上方放不下则折入框内顶部。
static bool draw_box_label(uint8_t* y, uint8_t* uv, int w, int h, int step_y, int step_uv,
                           cudaStream_t s, float box_x, float box_y, const std::string& text,
                           const uint8_t* rgb, int font_px, const std::string& font_path) {
    constexpr int kOverlap = 2;   // 底边压住 2px 顶边框，消除间隙
    if (blit_sprite_above_box(y, uv, w, h, step_y, step_uv, s, text, rgb, font_px, font_path,
                              box_x, box_y, kOverlap))
        return true;
    // 位图回退：按估算高度贴框顶
    const int th = 16 * label_bitmap_fs(font_px);
    int by = static_cast<int>(box_y) - th + kOverlap;
    if (by < 0) by = static_cast<int>(box_y) + 3;
    return draw_label_tag(y, uv, w, h, step_y, step_uv, s,
                          static_cast<int>(box_x), by, text, rgb, font_px, font_path);
}


struct Nv12View {
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    int w = 0, h = 0, step_y = 0, step_uv = 0;
    explicit Nv12View(modeldeploy::vision::ImageData& f) {
        if (f.type() != MdImageType::NV12 || f.plane_count() < 2) return;
        const auto p0 = f.plane(0), p1 = f.plane(1);
        y = const_cast<uint8_t*>(p0.data);
        uv = const_cast<uint8_t*>(p1.data);
        w = f.width(); h = f.height();
        step_y = p0.step > 0 ? p0.step : w;
        step_uv = p1.step > 0 ? p1.step : w;
    }
    bool ok() const { return y != nullptr; }
};

// COCO-17 骨架(1-indexed,端点做 -1);limb/kpt 色索引入 PosePaletteRgb
constexpr int kPoseSkeleton[19][2] = {
    {16,14},{14,12},{17,15},{15,13},{12,13},{6,12},{7,13},{6,7},{6,8},{7,9},
    {8,10},{9,11},{2,3},{1,2},{1,3},{2,4},{3,5},{4,6},{5,7}
};
constexpr int kPoseLimbColor[19] = {9,9,9,9,7,7,7,0,0,0,0,0,16,16,16,16,16,16,16};
constexpr int kPoseKptColor[17] = {16,16,16,16,16,0,0,0,0,0,0,9,9,9,9,9,9};
// 20 色 pose 调色板(RGB 序,由 CPU BGR 转置)
constexpr uint8_t kPosePaletteRgb[20][3] = {
    {0,128,255},{51,153,255},{102,178,255},{0,230,230},{255,153,255},{255,204,153},
    {255,102,255},{255,51,255},{255,178,102},{255,153,51},{153,153,255},{102,102,255},
    {51,51,255},{153,255,153},{102,255,102},{51,255,51},{0,255,0},{255,0,0},
    {0,0,255},{255,255,255}
};
// MediaPipe 21 点骨架(0-indexed,20 条)
constexpr int kHandSkeleton[20][2] = {
    {1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},
    {9,13},{13,14},{14,15},{15,16},{13,17},{17,18},{18,19},{19,20},{0,17}
};
// 21 点 hand 调色板(RGB 序,每关键点一个颜色)
constexpr uint8_t kHandPaletteRgb[21][3] = {
    {255,0,0},{255,85,0},{255,170,0},{255,255,0},{170,255,0},{85,255,0},
    {0,255,0},{0,255,85},{0,255,170},{0,255,255},{0,170,255},{0,85,255},
    {0,0,255},{85,0,255},{170,0,255},{255,0,255},{255,0,170},{255,0,85},
    {255,0,0},{170,0,0},{0,255,0}
};

void draw_kpts(Nv12View& v, cudaStream_t s, const std::vector<modeldeploy::vision::Point3f>& kpts,
               const uint8_t (*palette)[3], const int* kpt_color,
               int nkpt_color, int radius, bool draw_lines,
               const int (*skeleton)[2], int nskeleton, const int* limb_color) {
    for (size_t j = 0; j < kpts.size(); ++j) {
        if (kpts[j].z < 0.5f) continue;
        int ci = kpt_color ? kpt_color[j % nkpt_color] % 20 : static_cast<int>(j) % 20;
        float xs[1] = { kpts[j].x }, ys[1] = { kpts[j].y };
        modeldeploy::vision::draw_points_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                             xs, ys, 1, palette[ci][0], palette[ci][1], palette[ci][2],
                             radius, s);
    }
    if (!draw_lines || !skeleton) return;
    for (int j = 0; j < nskeleton; ++j) {
        const int a = skeleton[j][0] - 1, b = skeleton[j][1] - 1;   // 骨骼表 1-indexed → 0-indexed
        if (a < 0 || b < 0 || a >= (int)kpts.size() || b >= (int)kpts.size()) continue;
        if (kpts[a].z < 0.5f || kpts[b].z < 0.5f) continue;
        const uint8_t* c = palette[(limb_color ? limb_color[j] : 0) % 20];
        modeldeploy::vision::draw_line_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                           kpts[a].x, kpts[a].y, kpts[b].x, kpts[b].y,
                           c[0], c[1], c[2], 2, s);
    }
}

void draw_kpts_hand(Nv12View& v, cudaStream_t s, const std::vector<modeldeploy::vision::Point3f>& kpts) {
    for (size_t j = 0; j < kpts.size(); ++j) {
        if (kpts[j].z < 0.5f) continue;
        const uint8_t* c = kHandPaletteRgb[j % 21];
        float xs[1] = { kpts[j].x }, ys[1] = { kpts[j].y };
        modeldeploy::vision::draw_points_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                             xs, ys, 1, c[0], c[1], c[2], 3, s);
    }
    for (int j = 0; j < 20; ++j) {
        const int a = kHandSkeleton[j][0], b = kHandSkeleton[j][1];
        if (a >= (int)kpts.size() || b >= (int)kpts.size()) continue;
        if (kpts[a].z < 0.5f || kpts[b].z < 0.5f) continue;
        const uint8_t* c = kHandPaletteRgb[a % 21];
        modeldeploy::vision::draw_line_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                           kpts[a].x, kpts[a].y, kpts[b].x, kpts[b].y,
                           c[0], c[1], c[2], 2, s);
    }
}
}  // namespace

namespace modeldeploy::vision {
    // 惰性创建并返回持久 CUDA stream（backend 生命周期内复用）
    static cudaStream_t get_persistent_stream(void** slot) {
        cudaStream_t s = static_cast<cudaStream_t>(*slot);
        if (!s) {
            // NonBlocking：不与默认流(stream 0)隐式同步。否则与 20 路解码默认流上的
            // 同步 D2D 拷贝串行化，批推理耗时膨胀 ~5×（实测 23ms→4.6ms/batch-8）。
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            *slot = s;
        }
        return s;
    }

    CudaProcessorBackend::~CudaProcessorBackend() {
        if (stream_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
            stream_ = nullptr;
        }
    }

    bool CudaProcessorBackend::crop(const ImageData& image, float x, float y,
                                    float w, float h, ImageData* out) {
        if (!out || image.type() != MdImageType::NV12 || image.plane_count() < 2) return false;
        const int iw = image.width(), ih = image.height();
        int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
        int x1 = static_cast<int>(x + w), y1 = static_cast<int>(y + h);
        if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
        if (x1 > iw) x1 = iw; if (y1 > ih) y1 = ih;
        if (x1 <= x0 || y1 <= y0) return false;
        x0 &= ~1; y0 &= ~1; x1 &= ~1; y1 &= ~1;   // UV 偶数对齐
        if (x1 <= x0 || y1 <= y0) return false;
        const int cw = x1 - x0, ch = y1 - y0;     // 偶数
        const auto py = image.plane(0);
        const auto puv = image.plane(1);
        const int step_y = py.step > 0 ? py.step : iw;
        const int step_uv = puv.step > 0 ? puv.step : iw;
        const size_t ybytes = static_cast<size_t>(cw) * ch;
        const size_t uvbytes = static_cast<size_t>(cw) * (ch / 2);  // 每行 cw 字节(交错)，ch/2 行
        // 复用输出缓冲池，消除每帧 cudaMalloc/cudaFree 的分配/释放开销。
        // 注意 CudaOutputBufferPool 为单缓冲：本调用持有期间不得再次 acquire，调用方须在
        // 下次 acquire（下一次 crop/preprocess）之前完成本次裁剪结果的消费。
        uint8_t* dbuf = reinterpret_cast<uint8_t*>(out_pool_.acquire(ybytes + uvbytes));
        if (!dbuf) return false;
        // 在持久 stream 上拷贝 + 同步，避免与后续消费该裁剪块的非阻塞 stream 产生跨流竞争
        // （同步默认流 D2D 拷贝在多次连续调用后可能与持久流 kernel 竞速）。
        cudaStream_t cstream = get_persistent_stream(&stream_);
        cudaMemcpy2DAsync(dbuf, static_cast<size_t>(cw),
                          py.data + static_cast<size_t>(y0) * step_y + x0,
                          static_cast<size_t>(step_y),
                          static_cast<size_t>(cw), static_cast<size_t>(ch),
                          cudaMemcpyDeviceToDevice, cstream);
        cudaMemcpy2DAsync(dbuf + ybytes, static_cast<size_t>(cw),
                          puv.data + static_cast<size_t>(y0 / 2) * step_uv + x0,
                          static_cast<size_t>(step_uv),
                          static_cast<size_t>(cw), static_cast<size_t>(ch / 2),
                          cudaMemcpyDeviceToDevice, cstream);
        cudaStreamSynchronize(cstream);
        // from_planes 空 owner = 借用（缓冲由 out_pool_ 持有，后端析构统一释放）
        ImageData::Plane pl[2] = {{dbuf, cw}, {dbuf + ybytes, cw}};
        *out = ImageData::from_planes(pl, 2, MdImageType::NV12, cw, ch, Device::GPU);
        return !out->empty();
    }

    bool CudaProcessorBackend::yolo_preprocess(const ImageData& image, Tensor* out,
                                               const std::vector<int>& dst_size,
                                               float pad_val, LetterBoxRecord* record) {
        return yolo_preprocess_cuda(image, out, dst_size, pad_val, record,
                                    get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::yolo_preprocess_nv12(const uint8_t* src_y, const uint8_t* src_uv,
                                                    const std::vector<int>& src_size,
                                                    int step_y, int step_uv, Tensor* out,
                                                    const std::vector<int>& dst_size,
                                                    float pad_val, LetterBoxRecord* record,
                                                    Device src_device) {
        if (src_device == Device::TPU) {
            MD_LOG_ERROR << "CudaProcessorBackend: NV12 src_device TPU not supported." << std::endl;
            return false;
        }
        // 内部用 cudaPointerGetAttributes 校验：CPU 走 H2D，GPU 内存零拷贝直接使用
        return yolo_preprocess_nv12_cuda(src_y, src_uv, src_size, step_y, step_uv,
                                         out, dst_size, pad_val, record,
                                         get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::scrfd_preprocess(const ImageData& image, Tensor* out,
                                                const std::vector<int>& dst_size,
                                                float pad_val, LetterBoxRecord* record) {
        return scrfd_preprocess_cuda(image, out, dst_size, pad_val, record,
                                     get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::scrfd_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                      const std::vector<int>& dst_size,
                                                      float pad_val,
                                                      std::vector<LetterBoxRecord>* records) {
        return scrfd_preprocess_batch_cuda(images, out, dst_size, pad_val, records,
                                           &out_pool_, get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::fused_preprocess_common(
        const ImageData& image, Tensor* out,
        const std::vector<int>& dst_size,
        float origin_x, float origin_y,
        float scale_x, float scale_y,
        const std::vector<float>& alpha,
        const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        // NV12 双平面源 → NV12 融合 kernel（一次 launch：crop/resize + YUV2BGR + norm→CHW）
        if (image.type() == MdImageType::NV12 && image.plane_count() >= 2) {
            return fused_preprocess_common_nv12_cuda(image.plane(0).data, image.plane(1).data,
                                              {image.width(), image.height()},
                                              image.plane(0).step, image.plane(1).step,
                                              out, dst_size,
                                              origin_x, origin_y, scale_x, scale_y,
                                              alpha, beta, swap_rb, pad_value,
                                              get_persistent_stream(&stream_), &out_pool_);
        }
        return fused_preprocess_common_cuda(image.plane(0).data, {image.width(), image.height()},
                                     out, dst_size,
                                     origin_x, origin_y, scale_x, scale_y,
                                     alpha, beta, swap_rb, pad_value,
                                     get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::yolo_preprocess_batch(const std::vector<ImageData>& images, Tensor* out,
                                                     const std::vector<int>& dst_size,
                                                     float pad_val,
                                                     std::vector<LetterBoxRecord>* records) {
        // NV12 帧（设备/host）→ NV12 融合 kernel：设备帧零 PCIe，host 帧聚合 H2D
        if (!images.empty()) {
            bool all_nv12 = true;
            for (const auto& im : images) {
                if (im.type() != MdImageType::NV12 || im.plane_count() < 2) {
                    all_nv12 = false;
                    break;
                }
            }
            if (all_nv12) {
                return yolo_preprocess_nv12_batch_cuda(images, out, dst_size, pad_val, records,
                                                       get_persistent_stream(&stream_), &out_pool_);
            }
        }
        return yolo_preprocess_batch_cuda(images, out, dst_size, pad_val, records,
                                          get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::fused_preprocess_common_batch(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<int>& dst_size,
        const std::vector<float>& origins_x, const std::vector<float>& origins_y,
        const std::vector<float>& scales_x, const std::vector<float>& scales_y,
        const std::vector<float>& alpha, const std::vector<float>& beta,
        bool swap_rb, float pad_value) {
        // 全 NV12 批（device/host 混合亦可）→ 单次 NV12 融合 kernel
        if (!images.empty()) {
            bool all_nv12 = true;
            for (const auto& im : images) {
                if (im.type() != MdImageType::NV12 || im.plane_count() < 2) {
                    all_nv12 = false;
                    break;
                }
            }
            if (all_nv12) {
                return fused_preprocess_common_nv12_batch_cuda(images, out, dst_size,
                                                        origins_x, origins_y,
                                                        scales_x, scales_y,
                                                        alpha, beta, swap_rb, pad_value,
                                                        get_persistent_stream(&stream_), &out_pool_);
            }
        }
        return fused_preprocess_common_batch_cuda(images, out, dst_size, origins_x, origins_y,
                                           scales_x, scales_y, alpha, beta, swap_rb, pad_value,
                                           get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::ocr_det_preprocess(
        const std::vector<ImageData>& images, Tensor* out,
        const std::vector<std::array<int, 2>>& resize_sizes,
        const std::vector<int>& dst_size,
        const std::vector<float>& mean, const std::vector<float>& std,
        float pad_value) {
        const float alpha[3] = {
            1.0f / 255.0f / std[0],
            1.0f / 255.0f / std[1],
            1.0f / 255.0f / std[2]
        };
        const float beta[3] = {
            -mean[0] / std[0],
            -mean[1] / std[1],
            -mean[2] / std[2]
        };
        const float pad[3] = {
            pad_value * alpha[0] + beta[0],
            pad_value * alpha[1] + beta[1],
            pad_value * alpha[2] + beta[2]
        };
        return ocr_det_preprocess_cuda(images, out, resize_sizes, dst_size,
                                std::vector<float>(alpha, alpha + 3),
                                std::vector<float>(beta, beta + 3), pad,
                                get_persistent_stream(&stream_), &out_pool_);
    }

    bool CudaProcessorBackend::draw_rect_nv12(ImageData& frame, float x, float y, float w, float h,
                                              float r, float g, float b, int thickness) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_rect_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, w, h,
                                  static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), thickness,
                                  get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_polygon_nv12(ImageData& frame, const std::vector<Point2f>& pts,
                                                 float r, float g, float b, int thickness) {
        std::vector<float> xs, ys;
        xs.reserve(pts.size()); ys.reserve(pts.size());
        for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_polygon_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                     frame.width(), frame.height(),
                                     pl0.step, pl1.step,
                                     xs.data(), ys.data(), static_cast<int>(xs.size()),
                                     static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                     static_cast<uint8_t>(b), thickness,
                                     get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_points_nv12(ImageData& frame, const std::vector<Point3f>& pts,
                                                float r, float g, float b, int radius) {
        std::vector<float> xs, ys;
        xs.reserve(pts.size()); ys.reserve(pts.size());
        for (const auto& p : pts) { xs.push_back(p.x); ys.push_back(p.y); }
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_points_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                    frame.width(), frame.height(),
                                    pl0.step, pl1.step,
                                    xs.data(), ys.data(), static_cast<int>(xs.size()),
                                    static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                    static_cast<uint8_t>(b), radius,
                                    get_persistent_stream(&stream_));
    }

    bool CudaProcessorBackend::draw_text_nv12(ImageData& frame, float x, float y,
                                              const std::string& text,
                                              float r, float g, float b, int font_size) {
        const auto pl0 = frame.plane(0);
        const auto pl1 = frame.plane(1);
        return draw_text_nv12_gpu(const_cast<uint8_t*>(pl0.data), const_cast<uint8_t*>(pl1.data),
                                  frame.width(), frame.height(),
                                  pl0.step, pl1.step,
                                  x, y, text.c_str(),
                                  static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                                  static_cast<uint8_t>(b), font_size,
                                  get_persistent_stream(&stream_));
    }

    // ── 设备侧高层可视化（几何类已实现；其余桩暂保留，后续任务替换）──
    bool CudaProcessorBackend::vis_det_nv12(ImageData& frame,
                                            const std::vector<DetectionResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            const int x0 = static_cast<int>(r.box.x), y0 = static_cast<int>(r.box.y);
            const int x1 = x0 + static_cast<int>(r.box.width), y1 = y0 + static_cast<int>(r.box.height);
            ok = fill_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    x0, y0, x1, y1, c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            std::string label = label_name(opt, r.label_id) + ": " + score_str(r.score);
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                r.box.x, r.box.y, label, c, kLabelFontPx, opt.font_path) && ok;
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_obb_nv12(ImageData& frame,
                                            const std::vector<ObbResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            // 旋转矩形 → 4 角点(OpenCV RotatedRect 约定:angle 为度数,绕中心逆时针)
            const double ang = r.rotated_box.angle * 3.14159265358979323846 / 180.0;
            const float cs = static_cast<float>(std::cos(ang));
            const float sn = static_cast<float>(std::sin(ang));
            const float hw = r.rotated_box.width * 0.5f, hh = r.rotated_box.height * 0.5f;
            const float cx = r.rotated_box.xc, cy = r.rotated_box.yc;
            float pts[4][2] = {
                {-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}
            };
            std::vector<Point2f> poly(4);
            for (int i = 0; i < 4; ++i) {
                poly[i].x = cx + pts[i][0] * cs - pts[i][1] * sn;
                poly[i].y = cy + pts[i][0] * sn + pts[i][1] * cs;
            }
            std::vector<float> xs(4), ys(4);
            for (int i = 0; i < 4; ++i) { xs[i] = poly[i].x; ys[i] = poly[i].y; }
            ok = fill_polygon_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                       xs.data(), ys.data(), 4,
                                       c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_polygon_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                       xs.data(), ys.data(), 4,
                                       c[0], c[1], c[2], 2, s) && ok;
            std::string label = std::to_string(r.label_id) + ": " + score_str(r.score);
            const float obb_top = *std::min_element(ys.begin(), ys.end());
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                poly[1].x, obb_top, label, c, kLabelFontPx, opt.font_path) && ok;
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_pose_nv12(ImageData& frame,
                                             const std::vector<KeyPointsResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            ok = fill_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    (int)r.box.x, (int)r.box.y,
                                    (int)(r.box.x + r.box.width), (int)(r.box.y + r.box.height),
                                    c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                r.box.x, r.box.y, label, c, kLabelFontPx, opt.font_path) && ok;
            draw_kpts(v, s, r.keypoints, kPosePaletteRgb, kPoseKptColor, 17, 3, true,
                      kPoseSkeleton, 19, kPoseLimbColor);
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_keypoints_nv12(ImageData& frame,
                                                  const std::vector<KeyPointsResult>& result,
                                                  const VisionProcessorBackend::VisOptions& opt,
                                                  bool draw_lines) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            ok = fill_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    (int)r.box.x, (int)r.box.y,
                                    (int)(r.box.x + r.box.width), (int)(r.box.y + r.box.height),
                                    c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                r.box.x, r.box.y, label, c, kLabelFontPx, opt.font_path) && ok;
            draw_kpts(v, s, r.keypoints, kPosePaletteRgb, nullptr, 1, 3, draw_lines, nullptr, 0, nullptr);
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_hand_nv12(ImageData& frame,
                                             const std::vector<KeyPointsResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            ok = fill_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    (int)r.box.x, (int)r.box.y,
                                    (int)(r.box.x + r.box.width), (int)(r.box.y + r.box.height),
                                    c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            std::string label = "score: " + score_str(r.score);
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                r.box.x, r.box.y, label, c, kLabelFontPx, opt.font_path) && ok;
            draw_kpts_hand(v, s, r.keypoints);
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_ocr_nv12(ImageData& frame, const OCRResult& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (size_t i = 0; i < result.boxes.size(); ++i) {
            const auto& b = result.boxes[i];
            std::vector<float> xs(4), ys(4);
            for (int k = 0; k < 4; ++k) { xs[k] = static_cast<float>(b[k * 2]); ys[k] = static_cast<float>(b[k * 2 + 1]); }
            ok = fill_polygon_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                       xs.data(), ys.data(), 4, 66, 135, 245, alpha, s) && ok;
            ok = draw_polygon_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                       xs.data(), ys.data(), 4, 66, 135, 245, 2, s) && ok;
            if (i < result.text.size() && !result.text[i].empty()) {
                static const uint8_t kOcrBlue[3] = {66, 135, 245};
                const float top = *std::min_element(ys.begin(), ys.end());
                ok = blit_sprite_above_box(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                           result.text[i], kOcrBlue, kLabelFontPx, opt.font_path,
                                           xs[0], top, 2) && ok;
            }
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_lpr_nv12(ImageData& frame,
                                            const std::vector<LprResult>& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        const float alpha = static_cast<float>(opt.alpha);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            ok = fill_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    (int)r.box.x, (int)r.box.y,
                                    (int)(r.box.x + r.box.width), (int)(r.box.y + r.box.height),
                                    c[0], c[1], c[2], alpha, s) && ok;
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            std::string label = r.car_plate_str + " " + r.car_plate_color + " " + score_str(r.score);
            ok = draw_box_label(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                r.box.x, r.box.y, label, c, kLabelFontPx, opt.font_path) && ok;
            std::vector<float> xs, ys;
            for (const auto& p : r.keypoints) { xs.push_back(p.x); ys.push_back(p.y); }
            if (!xs.empty())
                ok = draw_points_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                          xs.data(), ys.data(), (int)xs.size(),
                                          0, 255, 0, 3, s) && ok;
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_attr_nv12(ImageData& frame,
                                             const std::vector<AttributeResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt,
                                             const std::vector<int>& abnormal_ids,
                                             bool show_attr) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        bool ok = true;
        std::unordered_map<int, bool> abnormal;
        for (int id : abnormal_ids) abnormal[id] = true;
        int obj_idx = 0;
        for (const auto& r : result) {
            const bool is_abnormal = abnormal.count(obj_idx) > 0 && abnormal[obj_idx];
            // 颜色语义对齐 CPU vis_attr.cpp:OpenCV BGR 下 Scalar(0,0,255)=红 → abnormal;
            // Scalar(0,255,0)=绿 → 正常。故数组(RGB 序)abnormal=红(255,0,0)、正常=绿(0,255,0)。
            uint8_t c[3];
            if (is_abnormal) {
                c[0] = 255; c[1] = 0; c[2] = 0;      // 红
            } else {
                c[0] = 0; c[1] = 255; c[2] = 0;      // 绿
            }
            if (r.box_score >= opt.threshold) {
                ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                        r.box.x, r.box.y, r.box.width, r.box.height,
                                        c[0], c[1], c[2], 2, s) && ok;
                if (show_attr) {
                    const int th = kLabelFontPx;
                    for (size_t i = 0; i < r.attr_scores.size(); ++i) {
                        std::string a = label_name(opt, static_cast<int>(i)) + ": " + score_str(r.attr_scores[i]);
                        const int by = static_cast<int>(r.box.y) + 2 + static_cast<int>(i) * (th + 4);
                        ok = draw_label_tag(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                            static_cast<int>(r.box.x) + 2, by, a, c,
                                            kLabelFontPx, opt.font_path) && ok;
                    }
                }
            }
            ++obj_idx;
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_cls_nv12(ImageData& frame, const ClassifyResult& result,
                                            const VisionProcessorBackend::VisOptions& opt,
                                            int top_k) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        if (top_k <= 0) top_k = 1;
        const int margin = 5;
        bool ok = true;
        int drawn = 0;
        const size_t n = std::min(result.label_ids.size(), result.scores.size());
        for (size_t i = 0; i < n && drawn < top_k; ++i) {
            if (result.scores[i] < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(result.label_ids[i])];
            const int y = margin + drawn * (kLabelFontPx + 4);
            std::string label = std::to_string(result.label_ids[i]) + ": " + score_str(result.scores[i]);
            ok = draw_label_tag(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv, s,
                                margin, y, label, c, kLabelFontPx, opt.font_path) && ok;
            ++drawn;
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_iseg_nv12(ImageData& frame,
                                             const std::vector<InstanceSegResult>& result,
                                             const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        bool ok = true;
        for (const auto& r : result) {
            if (r.score < opt.threshold) continue;
            const uint8_t* c = kClassPalette[palette_idx(r.label_id)];
            ok = draw_rect_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                    r.box.x, r.box.y, r.box.width, r.box.height,
                                    c[0], c[1], c[2], 2, s) && ok;
            if (r.mask.shape.size() == 2) {
                const size_t mh = static_cast<size_t>(r.mask.shape[0]);
                const size_t mw = static_cast<size_t>(r.mask.shape[1]);
                if (mw > 0 && mh > 0 && r.mask.buffer.size() >= mw * mh) {
                    uint8_t* d = nullptr;
                    if (cudaMalloc(&d, mw * mh) == cudaSuccess) {
                        if (cudaMemcpyAsync(d, r.mask.buffer.data(), mw * mh,
                                            cudaMemcpyHostToDevice, s) == cudaSuccess)
                            ok = overlay_mask_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                                       static_cast<int>(r.box.x),
                                                       static_cast<int>(r.box.y),
                                                       static_cast<int>(r.box.width),
                                                       static_cast<int>(r.box.height),
                                                       d, static_cast<int>(mw), static_cast<int>(mh),
                                                       c[0], c[1], c[2], 0.5f, s) && ok;
                        cudaFree(d);
                    }
                }
            }
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_sem_nv12(ImageData& frame, const SemSegResult& result,
                                            const VisionProcessorBackend::VisOptions& opt) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        if (result.shape.size() != 2) return false;
        const size_t h = static_cast<size_t>(result.shape[0]);
        const size_t w = static_cast<size_t>(result.shape[1]);
        if (w == 0 || h == 0 || result.labels.size() < w * h) return false;
        cudaStream_t s = get_persistent_stream(&stream_);
        uint8_t* d_labels = nullptr;
        bool ok = false;
        const size_t n = w * h;
        if (cudaMalloc(&d_labels, n) == cudaSuccess) {
            if (cudaMemcpyAsync(d_labels, result.labels.data(), n, cudaMemcpyHostToDevice, s) == cudaSuccess)
                ok = overlay_labels_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                             d_labels, static_cast<int>(w), static_cast<int>(h),
                                             static_cast<float>(opt.alpha), s);
            cudaFree(d_labels);
        }
        return ok;
    }
    bool CudaProcessorBackend::vis_depth_nv12(ImageData& frame, const DepthResult& result,
                                              const VisionProcessorBackend::VisOptions& opt,
                                              bool colorize) {
        Nv12View v(frame);
        if (!v.ok()) return false;
        if (result.shape.size() != 2) return false;
        const size_t h = static_cast<size_t>(result.shape[0]);
        const size_t w = static_cast<size_t>(result.shape[1]);
        const size_t n = w * h;
        if (w == 0 || h == 0 || result.depth.size() < n) return false;
        float mn = result.depth[0], mx = result.depth[0];
        for (size_t i = 0; i < n; ++i) {
            if (result.depth[i] < mn) mn = result.depth[i];
            if (result.depth[i] > mx) mx = result.depth[i];
        }
        const float range = (mx - mn) > 1e-6f ? (mx - mn) : 1.0f;
        std::vector<uint8_t> d8(n);
        for (size_t i = 0; i < n; ++i)
            d8[i] = static_cast<uint8_t>((result.depth[i] - mn) / range * 255.0f);
        cudaStream_t s = get_persistent_stream(&stream_);
        uint8_t* d = nullptr;
        bool ok = false;
        if (cudaMalloc(&d, n) == cudaSuccess) {
            if (cudaMemcpyAsync(d, d8.data(), n, cudaMemcpyHostToDevice, s) == cudaSuccess)
                ok = overlay_depth_nv12_gpu(v.y, v.uv, v.w, v.h, v.step_y, v.step_uv,
                                            d, static_cast<int>(w), static_cast<int>(h),
                                            colorize, static_cast<float>(opt.alpha), s);
            cudaFree(d);
        }
        return ok;
    }
} // namespace modeldeploy::vision
