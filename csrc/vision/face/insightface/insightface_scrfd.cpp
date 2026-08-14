//
// insightface buffalo_l det_10g：SCRFD 人脸检测实现。
// 前处理用 fused_preprocess（CPU SIMD/CUDA/BMCV 多后端），推理走 Runtime，解码为独立 postprocessor。
//
#include "core/md_log.h"
#include "vision/face/insightface/insightface_scrfd.h"
#include "vision/utils.h"
#include "core/tensor.h"
#include <algorithm>
#include <cstring>

namespace modeldeploy::vision::face {

    // ==================== Preprocessor ====================

    InsightFaceDetPreprocessor::InsightFaceDetPreprocessor() {
        size_ = {640, 640};
    }

    bool InsightFaceDetPreprocessor::run(const ImageData& image, Tensor* output,
                                         LetterBoxRecord* letter_box_record) const {
        // 与 python SCRFD._detect_candidates 一致：等比例双线性 resize + 左上放置 + pad 0
        const int src_w = image.width();
        const int src_h = image.height();
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        const float im_ratio = static_cast<float>(src_h) / src_w;
        const float model_ratio = static_cast<float>(dst_h) / dst_w;
        int new_w, new_h;
        if (im_ratio > model_ratio) {
            new_h = dst_h;
            new_w = static_cast<int>(std::round(new_h / im_ratio));
        } else {
            new_w = dst_w;
            new_h = static_cast<int>(std::round(new_w * im_ratio));
        }
        // 双线性 resize + 左上放置（python cv2.resize INTER_LINEAR + det_img 左上）
        cv::Mat src_mat;
        image.to_mat(src_mat);
        cv::Mat resized;
        cv::resize(src_mat, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat det_img(dst_h, dst_w, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));
        // 记录 det_scale（后处理缩放回原图）
        letter_box_record->ipt_w = static_cast<float>(src_w);
        letter_box_record->ipt_h = static_cast<float>(src_h);
        letter_box_record->scale = static_cast<float>(new_h) / src_h;
        letter_box_record->pad_w = 0.0f;
        letter_box_record->pad_h = 0.0f;
        letter_box_record->out_w = static_cast<float>(dst_w);
        letter_box_record->out_h = static_cast<float>(dst_h);
        // blob：(x-127.5)/128 + BGR2RGB，与 cv2.dnn.blobFromImage 一致
        // 手写 blob（OpenCV 5 预编译包无 dnn 模块）
        std::vector<float> blob(static_cast<size_t>(3) * dst_h * dst_w);
        const uint8_t* src = det_img.data;
        for (int c = 0; c < 3; ++c) {
            const int src_c = 2 - c; // swapRB: 输出通道0=R(原[2])
            float* plane = blob.data() + static_cast<size_t>(c) * dst_h * dst_w;
            for (int i = 0; i < dst_h * dst_w; ++i) {
                plane[i] = (static_cast<float>(src[i * 3 + src_c]) - 127.5f) * (1.0f / 128.0f);
            }
        }
        // 拷贝进 tensor（不共享局部 blob 生命周期）
        output->allocate({1, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), blob.data(), blob.size() * sizeof(float));
        return true;
    }

    bool InsightFaceDetPreprocessor::run(const std::vector<ImageData>& images, Tensor* output,
                                         std::vector<LetterBoxRecord>* letter_box_records) const {
        if (images.empty()) return false;
        letter_box_records->resize(images.size());
        if (images.size() == 1) {
            return run(images[0], output, &(*letter_box_records)[0]);
        }
        // 整批：逐图双线性预处理 + 拼接 batch tensor
        const int n = static_cast<int>(images.size());
        const int dst_w = size_[0];
        const int dst_h = size_[1];
        std::vector<float> batch_blob(static_cast<size_t>(n) * 3 * dst_h * dst_w);
        for (int i = 0; i < n; ++i) {
            Tensor single;
            if (!run(images[i], &single, &(*letter_box_records)[i])) return false;
            std::memcpy(batch_blob.data() + static_cast<size_t>(i) * 3 * dst_h * dst_w,
                        single.data(), static_cast<size_t>(3) * dst_h * dst_w * sizeof(float));
        }
        output->allocate({n, 3, dst_h, dst_w}, DataType::FP32, Device::CPU);
        std::memcpy(output->data(), batch_blob.data(), batch_blob.size() * sizeof(float));
        return true;
    }

    // ==================== Postprocessor ====================

    namespace {
        constexpr int kFmc = 3;
        constexpr int kFeatStrideFpn[3] = {8, 16, 32};
        constexpr int kNumAnchors = 2;

        // 生成 anchor centers：交错布局（python np.stack([centers]*2, axis=1).reshape(-1,2)）
        std::vector<float> gen_anchor_centers(int height, int width, int stride) {
            const int K = height * width;
            std::vector<float> base(static_cast<size_t>(K) * 2);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int idx = y * width + x;
                    base[idx * 2] = static_cast<float>(x * stride);
                    base[idx * 2 + 1] = static_cast<float>(y * stride);
                }
            std::vector<float> dup(static_cast<size_t>(K) * kNumAnchors * 2);
            for (int i = 0; i < K; ++i)
                for (int a = 0; a < kNumAnchors; ++a) {
                    dup[(i * kNumAnchors + a) * 2] = base[i * 2];
                    dup[(i * kNumAnchors + a) * 2 + 1] = base[i * 2 + 1];
                }
            return dup;
        }

        // 标准 NMS（与 python 一致，IoU <= thresh 保留）
        std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                             const std::vector<float>& scores, float thresh) {
            const int n = static_cast<int>(boxes.size());
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            std::vector<int> keep;
            std::vector<bool> removed(n, false);
            for (int oi = 0; oi < n; ++oi) {
                const int i = order[oi];
                if (removed[i]) continue;
                keep.push_back(i);
                for (int oj = oi + 1; oj < n; ++oj) {
                    const int j = order[oj];
                    if (removed[j]) continue;
                    const float x1 = std::max(boxes[i][0], boxes[j][0]);
                    const float y1 = std::max(boxes[i][1], boxes[j][1]);
                    const float x2 = std::min(boxes[i][2], boxes[j][2]);
                    const float y2 = std::min(boxes[i][3], boxes[j][3]);
                    const float w = std::max(0.0f, x2 - x1 + 1);
                    const float h = std::max(0.0f, y2 - y1 + 1);
                    const float inter = w * h;
                    const float area_i = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1);
                    const float area_j = (boxes[j][2] - boxes[j][0] + 1) * (boxes[j][3] - boxes[j][1] + 1);
                    const float ovr = inter / (area_i + area_j - inter);
                    if (ovr > thresh) removed[j] = true;
                }
            }
            return keep;
        }
    } // namespace

    bool InsightFaceDetPostprocessor::run(const std::vector<Tensor>& infer_results,
                                          const std::vector<LetterBoxRecord>& letter_box_records,
                                          const std::vector<float>& det_scales,
                                          std::vector<std::vector<InsightFaceBox>>* results) {
        results->resize(letter_box_records.size());
        // 输入尺寸（640x640）从输出 shape 推导
        // 每个 batch 的图共享输入尺寸
        // 输出顺序：3 score, 3 bbox, 3 kps
        const size_t batch = letter_box_records.size();
        // 单图 batch=1
        for (size_t b = 0; b < batch; ++b) {
            const float det_scale = det_scales[b];
            const int dst_h = 640;
            const int dst_w = 640;
            auto& out = (*results)[b];
            out.clear();

            std::vector<float> scores_list, bboxes_list, kpss_list;
            for (int idx = 0; idx < kFmc; ++idx) {
                const int stride = kFeatStrideFpn[idx];
                const float* scores_ptr = static_cast<const float*>(infer_results[idx].data());
                const float* bbox_ptr = static_cast<const float*>(infer_results[idx + kFmc].data());
                const float* kps_ptr = static_cast<const float*>(infer_results[idx + kFmc * 2].data());
                const int H = dst_h / stride;
                const int W = dst_w / stride;
                const int K = H * W;
                const int total = K * kNumAnchors;

                // bbox/kps 乘 stride
                std::vector<float> bbox_scaled(static_cast<size_t>(total) * 4);
                std::vector<float> kps_scaled(static_cast<size_t>(total) * 10);
                for (int i = 0; i < total; ++i) {
                    for (int j = 0; j < 4; ++j) bbox_scaled[i * 4 + j] = bbox_ptr[i * 4 + j] * stride;
                    for (int j = 0; j < 10; ++j) kps_scaled[i * 10 + j] = kps_ptr[i * 10 + j] * stride;
                }
                const auto centers = gen_anchor_centers(H, W, stride);
                const int n_centers = static_cast<int>(centers.size() / 2);

                std::vector<int> pos_inds;
                for (int i = 0; i < total; ++i)
                    if (scores_ptr[i] >= 0.5f) pos_inds.push_back(i);

                // distance2bbox
                std::vector<float> bboxes(static_cast<size_t>(total) * 4);
                for (int i = 0; i < n_centers; ++i) {
                    const float cx = centers[i * 2], cy = centers[i * 2 + 1];
                    bboxes[i * 4 + 0] = cx - bbox_scaled[i * 4 + 0];
                    bboxes[i * 4 + 1] = cy - bbox_scaled[i * 4 + 1];
                    bboxes[i * 4 + 2] = cx + bbox_scaled[i * 4 + 2];
                    bboxes[i * 4 + 3] = cy + bbox_scaled[i * 4 + 3];
                }
                // distance2kps
                std::vector<float> kpss(static_cast<size_t>(total) * 10);
                for (int i = 0; i < n_centers; ++i) {
                    const float cx = centers[i * 2], cy = centers[i * 2 + 1];
                    for (int j = 0; j < 10; j += 2) {
                        kpss[i * 10 + j] = cx + kps_scaled[i * 10 + j];
                        kpss[i * 10 + j + 1] = cy + kps_scaled[i * 10 + j + 1];
                    }
                }
                for (int p : pos_inds) {
                    scores_list.push_back(scores_ptr[p]);
                    bboxes_list.insert(bboxes_list.end(), bboxes.begin() + p * 4, bboxes.begin() + p * 4 + 4);
                    kpss_list.insert(kpss_list.end(), kpss.begin() + p * 10, kpss.begin() + p * 10 + 10);
                }
            }

            if (scores_list.empty()) continue;
            // 排序 + 缩放回原图 + NMS
            const int n_det = static_cast<int>(scores_list.size());
            std::vector<int> order(n_det);
            for (int i = 0; i < n_det; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return scores_list[a] > scores_list[b]; });

            std::vector<std::array<float, 4>> boxes_in;
            std::vector<float> scores_in;
            std::vector<float> kpss_in(static_cast<size_t>(n_det) * 10);
            for (int i = 0; i < n_det; ++i) {
                const int o = order[i];
                boxes_in.push_back({bboxes_list[o * 4] / det_scale, bboxes_list[o * 4 + 1] / det_scale,
                                    bboxes_list[o * 4 + 2] / det_scale, bboxes_list[o * 4 + 3] / det_scale});
                scores_in.push_back(scores_list[o]);
                for (int j = 0; j < 10; ++j) kpss_in[i * 10 + j] = kpss_list[o * 10 + j] / det_scale;
            }
            const auto keep = nms(boxes_in, scores_in, nms_thresh_);
            out.reserve(keep.size());
            for (int idx : keep) {
                InsightFaceBox bb;
                bb.bbox = boxes_in[idx];
                bb.score = scores_in[idx];
                for (int j = 0; j < 5; ++j) bb.kps.push_back({kpss_in[idx * 10 + j * 2], kpss_in[idx * 10 + j * 2 + 1]});
                out.push_back(std::move(bb));
            }
        }
        return true;
    }

    // ==================== Model ====================

    InsightFaceDet::InsightFaceDet(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    bool InsightFaceDet::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        preprocessor_.set_processor_backend(
            create_processor_backend(runtime_option.device, runtime_option.backend,
                                     runtime_option.device_id));
        return true;
    }

    bool InsightFaceDet::predict(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                                 TimerArray* timers) {
        std::vector<std::vector<InsightFaceBox>> results;
        if (!batch_predict({image}, &results, timers)) return false;
        *boxes = std::move(results[0]);
        return true;
    }

    bool InsightFaceDet::batch_predict(const std::vector<ImageData>& images,
                                       std::vector<std::vector<InsightFaceBox>>* boxes,
                                       TimerArray* timers) {
        std::vector<ImageData> _images = images;
        std::vector<LetterBoxRecord> lbrs;
        std::vector<float> det_scales;
        if (timers) timers->pre_timer.start();
        reused_input_tensors_.resize(1);
        if (!preprocessor_.run(_images, &reused_input_tensors_[0], &lbrs)) {
            MD_LOG_ERROR << "Failed to preprocess." << std::endl;
            return false;
        }
        if (timers) timers->pre_timer.stop();
        reused_input_tensors_[0].set_name(get_input_info(0).name);
        if (timers) timers->infer_timer.start();
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();
        // det_scale 从 preprocessor 记录
        for (const auto& l : lbrs) det_scales.push_back(l.scale);
        if (timers) timers->post_timer.start();
        if (!postprocessor_.run(reused_output_tensors_, lbrs, det_scales, boxes)) return false;
        if (timers) timers->post_timer.stop();
        return true;
    }

    std::unique_ptr<InsightFaceDet> InsightFaceDet::clone() const {
        auto clone_model = std::make_unique<InsightFaceDet>(
            runtime_option.model_file, runtime_option);
        clone_model->set_runtime(clone_model->clone_runtime());
        clone_model->preprocessor_ = preprocessor_;
        clone_model->postprocessor_ = postprocessor_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

} // namespace modeldeploy::vision::face
