//
// insightface buffalo_l det_10g SCRFD 实现。
// 与 python insightface.model_zoo.scrfd.SCRFD 逐值对齐：
//   - 预处理：blobFromImage(img, 1/128, input_size, mean=127.5, swapRB=True)
//   - 解码：distance2bbox / distance2kps，anchor centers = mgrid * stride
//   - 后处理：score 阈值过滤 + 缩放回原图 + NMS
//
#include "core/md_log.h"
#include "vision/face/insightface/insightface_scrfd.h"
#include "vision/face/insightface/face_align_utils.h"
#include "core/tensor.h"
#include "vision/utils.h"

namespace modeldeploy::vision::face {

    namespace {
        constexpr int kFmc = 3;
        constexpr int kFeatStrideFpn[3] = {8, 16, 32};
        constexpr int kNumAnchors = 2;
        constexpr float kInputMean = 127.5f;
        constexpr float kInputStd = 128.0f;
    } // namespace

    InsightFaceDet::InsightFaceDet(const std::string& model_file,
                                   const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = Initialize();
    }

    bool InsightFaceDet::Initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "Failed to initialize modeldeploy runtime." << std::endl;
            return false;
        }
        return true;
    }

    namespace {
        // 距离转 bbox（与 python distance2bbox 一致）
        inline void distance2bbox(const float* centers, const float* dist,
                                  const int n, float* boxes) {
            for (int i = 0; i < n; ++i) {
                const float cx = centers[i * 2];
                const float cy = centers[i * 2 + 1];
                boxes[i * 4 + 0] = cx - dist[i * 4 + 0];
                boxes[i * 4 + 1] = cy - dist[i * 4 + 1];
                boxes[i * 4 + 2] = cx + dist[i * 4 + 2];
                boxes[i * 4 + 3] = cy + dist[i * 4 + 3];
            }
        }

        // 距离转 kps（与 python distance2kps 一致）
        inline void distance2kps(const float* centers, const float* dist,
                                 const int n, const int kps_dim, float* kpss) {
            // centers: n x 2, dist: n x 10, kpss: n x 10
            for (int i = 0; i < n; ++i) {
                const float cx = centers[i * 2];
                const float cy = centers[i * 2 + 1];
                for (int j = 0; j < kps_dim; j += 2) {
                    kpss[i * kps_dim + j] = cx + dist[i * kps_dim + j];
                    kpss[i * kps_dim + j + 1] = cy + dist[i * kps_dim + j + 1];
                }
            }
        }

        // 生成 anchor centers（与 python mgrid + anchor stack 一致）
        // python: centers = mgrid...reshape(-1,2); 然后 stack([centers]*num_anchors, axis=1).reshape(-1,2)
        // => 交错：pos 0,1 同 center0; pos 2,3 同 center1...
        // 返回 height*width*num_anchors x 2
        std::vector<float> gen_anchor_centers(int height, int width, int stride) {
            const int K = height * width;
            std::vector<float> base(static_cast<size_t>(K) * 2);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int idx = y * width + x;
                    base[idx * 2] = static_cast<float>(x * stride);
                    base[idx * 2 + 1] = static_cast<float>(y * stride);
                }
            }
            std::vector<float> dup(static_cast<size_t>(K) * kNumAnchors * 2);
            for (int i = 0; i < K; ++i)
                for (int a = 0; a < kNumAnchors; ++a) {
                    dup[(i * kNumAnchors + a) * 2] = base[i * 2];
                    dup[(i * kNumAnchors + a) * 2 + 1] = base[i * 2 + 1];
                }
            return dup;
        }

        // 标准 NMS（与 python nms 一致，IoU <= thresh 保留）
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

    bool InsightFaceDet::predict(const ImageData& image, std::vector<InsightFaceBox>* boxes,
                                 TimerArray* timers) {
        if (!image.data() || !boxes) return false;
        boxes->clear();

        const int src_w = image.width();
        const int src_h = image.height();
        const int dst_w = input_size_[0];
        const int dst_h = input_size_[1];

        // 与 python _detect_candidates 一致：等比例缩放 + 右下填充
        // im_ratio = h/w, model_ratio = dst_h/dst_w
        const float im_ratio = static_cast<float>(src_h) / src_w;
        const float model_ratio = static_cast<float>(dst_h) / dst_w;
        int new_h, new_w;
        if (im_ratio > model_ratio) {
            new_h = dst_h;
            new_w = static_cast<int>(std::round(new_h / im_ratio));
        } else {
            new_w = dst_w;
            new_h = static_cast<int>(std::round(new_w * im_ratio));
        }
        const float det_scale = static_cast<float>(new_h) / src_h;

        // 缩放 + 填充到 (dst_h, dst_w)
        cv::Mat src_mat;
        image.to_mat(src_mat);
        cv::Mat resized;
        cv::resize(src_mat, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat det_img(dst_h, dst_w, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));

        // 预处理：blobFromImage(1/128, mean=127.5, swapRB=True)
        // (input - 127.5) / 128, BGR->RGB
        cv::Mat blob = make_blob_from_image(det_img, 1.0f / kInputStd,
                                            cv::Scalar(kInputMean, kInputMean, kInputMean),
                                            true /*swapRB*/);

        // 推理
        std::vector<Tensor> input_tensors(1);
        // blob 是 NCHW float32，包成 Tensor（零拷贝共享）
        input_tensors[0].from_external_memory(blob.data, {1, 3, dst_h, dst_w},
                                              DataType::FP32, nullptr, Device::CPU,
                                              get_input_info(0).name);
        std::vector<Tensor> output_tensors;
        if (timers) timers->infer_timer.start();
        if (!infer(input_tensors, &output_tensors)) {
            MD_LOG_ERROR << "Failed to inference." << std::endl;
            return false;
        }
        if (timers) timers->infer_timer.stop();

        // 解码（与 python forward 一致）
        // net_outs: [score(3), bbox(3), kps(3)]，按 stride 顺序
        std::vector<float> all_scores, all_boxes, all_kpss;
        std::vector<int> all_inds;
        // 各 stride 位置：scores_list, bboxes_list, kpss_list
        std::vector<float> scores_list, bboxes_list, kpss_list;
        std::vector<int> scores_cnt_list, bboxes_cnt_list, kpss_cnt_list;

        for (int idx = 0; idx < kFmc; ++idx) {
            const int stride = kFeatStrideFpn[idx];
            const float* scores_ptr = static_cast<const float*>(output_tensors[idx].data());
            const float* bbox_ptr = static_cast<const float*>(output_tensors[idx + kFmc].data());
            const float* kps_ptr = static_cast<const float*>(output_tensors[idx + kFmc * 2].data());
            const int H = dst_h / stride;
            const int W = dst_w / stride;
            const int K = H * W;
            const int total = K * kNumAnchors;

            // bbox/kps 乘 stride
            std::vector<float> bbox_scaled(static_cast<size_t>(total) * 4);
            std::vector<float> kps_scaled(static_cast<size_t>(total) * 10);
            for (int i = 0; i < total; ++i) {
                bbox_scaled[i * 4 + 0] = bbox_ptr[i * 4 + 0] * stride;
                bbox_scaled[i * 4 + 1] = bbox_ptr[i * 4 + 1] * stride;
                bbox_scaled[i * 4 + 2] = bbox_ptr[i * 4 + 2] * stride;
                bbox_scaled[i * 4 + 3] = bbox_ptr[i * 4 + 3] * stride;
                for (int j = 0; j < 10; ++j) kps_scaled[i * 10 + j] = kps_ptr[i * 10 + j] * stride;
            }

            const auto centers = gen_anchor_centers(H, W, stride);
            const int n_centers = static_cast<int>(centers.size() / 2);

            // score >= thresh 的索引
            std::vector<int> pos_inds;
            for (int i = 0; i < total; ++i) {
                if (scores_ptr[i] >= det_thresh_) pos_inds.push_back(i);
            }
            // bboxes = distance2bbox(centers, bbox_scaled)
            std::vector<float> bboxes(static_cast<size_t>(total) * 4);
            distance2bbox(centers.data(), bbox_scaled.data(), n_centers, bboxes.data());
            // kpss = distance2kps
            std::vector<float> kpss(static_cast<size_t>(total) * 10);
            distance2kps(centers.data(), kps_scaled.data(), n_centers, 10, kpss.data());

            // 收集 pos 索引
            for (int p : pos_inds) {
                scores_list.push_back(scores_ptr[p]);
                bboxes_list.insert(bboxes_list.end(), bboxes.begin() + p * 4, bboxes.begin() + p * 4 + 4);
                kpss_list.insert(kpss_list.end(), kpss.begin() + p * 10, kpss.begin() + p * 10 + 10);
            }
            scores_cnt_list.push_back(static_cast<int>(pos_inds.size()));
            bboxes_cnt_list.push_back(static_cast<int>(pos_inds.size()));
            kpss_cnt_list.push_back(static_cast<int>(pos_inds.size()));
        }

        if (scores_list.empty()) return true;

        // 排序（score 降序）
        const int n_det = static_cast<int>(scores_list.size());
        std::vector<int> order(n_det);
        for (int i = 0; i < n_det; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return scores_list[a] > scores_list[b]; });

        // 缩放回原图：/ det_scale
        // 注意：bboxes 的坐标是输入图（dst_w x dst_h）坐标，除 det_scale 得原图坐标
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

        // NMS
        const auto keep = nms(boxes_in, scores_in, nms_thresh_);

        boxes->reserve(keep.size());
        for (int idx : keep) {
            InsightFaceBox b;
            b.bbox = boxes_in[idx];
            b.score = scores_in[idx];
            for (int j = 0; j < 5; ++j) {
                b.kps.push_back({kpss_in[idx * 10 + j * 2], kpss_in[idx * 10 + j * 2 + 1]});
            }
            boxes->push_back(std::move(b));
        }
        return true;
    }

    std::unique_ptr<InsightFaceDet> InsightFaceDet::clone() const {
        auto clone_model = std::make_unique<InsightFaceDet>(*this);
        clone_model->set_runtime(clone_model->clone_runtime());
        return clone_model;
    }

} // namespace modeldeploy::vision::face
