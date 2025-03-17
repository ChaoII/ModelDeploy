//
// Created by aichao on 2025/2/21.
//

#include <map>
#include "csrc/vision/ocr/utils/clipper.h"
#include "csrc/vision/ocr/utils/ocr_postprocess_op.h"


namespace modeldeploy::vision::ocr {
    void PostProcessor::get_contour_area(const std::vector<std::vector<float>>& box,
                                         float unclip_ratio, float& distance) {
        constexpr int pts_num = 4;
        float area = 0.0f;
        float dist = 0.0f;
        for (int i = 0; i < pts_num; i++) {
            area += box[i][0] * box[(i + 1) % pts_num][1] -
                box[i][1] * box[(i + 1) % pts_num][0];
            dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                (box[i][0] - box[(i + 1) % pts_num][0]) +
                (box[i][1] - box[(i + 1) % pts_num][1]) *
                (box[i][1] - box[(i + 1) % pts_num][1]));
        }
        area = fabs(static_cast<float>(area / 2.0));
        distance = area * unclip_ratio / dist;
    }

    cv::RotatedRect PostProcessor::un_clip(std::vector<std::vector<float>> box,
                                           const float& unclip_ratio) {
        float distance = 1.0;
        get_contour_area(box, unclip_ratio, distance);
        ClipperLib::ClipperOffset offset;
        ClipperLib::Path p;
        p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
            << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
            << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
            << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
        offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
        ClipperLib::Paths soln;
        offset.Execute(soln, distance);
        std::vector<cv::Point2f> points;
        for (int j = 0; j < soln.size(); j++) {
            for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
                points.emplace_back(soln[j][i].X, soln[j][i].Y);
            }
        }
        cv::RotatedRect res;
        if (points.size() <= 0) {
            res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
        }
        else {
            res = cv::minAreaRect(points);
        }
        return res;
    }

    float** PostProcessor::mat2_vec(cv::Mat mat) {
        auto** array = new float*[mat.rows];
        for (int i = 0; i < mat.rows; ++i) array[i] = new float[mat.cols];
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                array[i][j] = mat.at<float>(i, j);
            }
        }

        return array;
    }

    std::vector<std::vector<int>> PostProcessor::order_points_clockwise(
        std::vector<std::vector<int>> pts) {
        std::vector<std::vector<int>> box = pts;
        std::sort(box.begin(), box.end(), x_sort_int);
        std::vector leftmost = {box[0], box[1]};
        std::vector rightmost = {box[2], box[3]};
        if (leftmost[0][1] > leftmost[1][1]) std::swap(leftmost[0], leftmost[1]);
        if (rightmost[0][1] > rightmost[1][1]) std::swap(rightmost[0], rightmost[1]);
        std::vector<std::vector<int>> rect = {
            leftmost[0], rightmost[0], rightmost[1],
            leftmost[1]
        };
        return rect;
    }

    std::vector<std::vector<float>> PostProcessor::mat2_vector(cv::Mat mat) {
        std::vector<std::vector<float>> img_vec;
        std::vector<float> tmp;
        for (int i = 0; i < mat.rows; ++i) {
            tmp.clear();
            for (int j = 0; j < mat.cols; ++j) {
                tmp.push_back(mat.at<float>(i, j));
            }
            img_vec.push_back(tmp);
        }
        return img_vec;
    }

    bool PostProcessor::x_sort_fp32(std::vector<float> a, std::vector<float> b) {
        if (a[0] != b[0]) return a[0] < b[0];
        return false;
    }

    bool PostProcessor::x_sort_int(std::vector<int> a, std::vector<int> b) {
        if (a[0] != b[0]) return a[0] < b[0];
        return false;
    }

    std::vector<std::vector<float>> PostProcessor::get_mini_boxes(cv::RotatedRect box,
                                                                  float& ssid) {
        ssid = std::max(box.size.width, box.size.height);
        cv::Mat points;
        cv::boxPoints(box, points);
        auto array = mat2_vector(points);
        std::sort(array.begin(), array.end(), x_sort_fp32);
        std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                           idx4 = array[3];
        if (array[3][1] <= array[2][1]) {
            idx2 = array[3];
            idx3 = array[2];
        }
        else {
            idx2 = array[2];
            idx3 = array[3];
        }
        if (array[1][1] <= array[0][1]) {
            idx1 = array[1];
            idx4 = array[0];
        }
        else {
            idx1 = array[0];
            idx4 = array[1];
        }

        array[0] = idx1;
        array[1] = idx2;
        array[2] = idx3;
        array[3] = idx4;
        return array;
    }

    float PostProcessor::polygon_score_acc(std::vector<cv::Point> contour,
                                           cv::Mat pred) {
        const int width = pred.cols;
        const int height = pred.rows;
        std::vector<float> box_x;
        std::vector<float> box_y;
        for (int i = 0; i < contour.size(); ++i) {
            box_x.push_back(contour[i].x);
            box_y.push_back(contour[i].y);
        }
        int xmin =
            clamp(static_cast<int>(std::floor(*std::min_element(box_x.begin(), box_x.end()))), 0,
                  width - 1);
        int xmax =
            clamp(static_cast<int>(std::ceil(*std::max_element(box_x.begin(), box_x.end()))), 0,
                  width - 1);
        int ymin =
            clamp(static_cast<int>(std::floor(*std::min_element(box_y.begin(), box_y.end()))), 0,
                  height - 1);
        int ymax =
            clamp(static_cast<int>(std::ceil(*std::max_element(box_y.begin(), box_y.end()))), 0,
                  height - 1);
        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        cv::Point* rook_point = new cv::Point[contour.size()];
        for (int i = 0; i < contour.size(); ++i) {
            rook_point[i] = cv::Point(static_cast<int>(box_x[i]) - xmin, static_cast<int>(box_y[i]) - ymin);
        }
        const cv::Point* ppt[1] = {rook_point};
        const int npt[] = {static_cast<int>(contour.size())};
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
        cv::Mat croppedImg;
        pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
            .copyTo(croppedImg);
        const float score = cv::mean(croppedImg, mask)[0];
        delete[] rook_point;
        return score;
    }

    float PostProcessor::box_score_fast(std::vector<std::vector<float>> box_array,
                                        cv::Mat pred) {
        const auto array = box_array;
        const int width = pred.cols;
        const int height = pred.rows;

        float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
        float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

        int xmin = clamp(static_cast<int>(std::floor(*std::min_element(box_x, box_x + 4))), 0,
                         width - 1);
        int xmax = clamp(static_cast<int>(std::ceil(*std::max_element(box_x, box_x + 4))), 0,
                         width - 1);
        int ymin = clamp(static_cast<int>(std::floor(*std::min_element(box_y, box_y + 4))), 0,
                         height - 1);
        int ymax = clamp(static_cast<int>(std::ceil(*std::max_element(box_y, box_y + 4))), 0,
                         height - 1);
        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        cv::Point root_point[4];
        root_point[0] = cv::Point(static_cast<int>(array[0][0]) - xmin, static_cast<int>(array[0][1]) - ymin);
        root_point[1] = cv::Point(static_cast<int>(array[1][0]) - xmin, static_cast<int>(array[1][1]) - ymin);
        root_point[2] = cv::Point(static_cast<int>(array[2][0]) - xmin, static_cast<int>(array[2][1]) - ymin);
        root_point[3] = cv::Point(static_cast<int>(array[3][0]) - xmin, static_cast<int>(array[3][1]) - ymin);
        const cv::Point* ppt[1] = {root_point};
        constexpr int npt[] = {4};
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
        cv::Mat croppedImg;
        pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
            .copyTo(croppedImg);
        const auto score = cv::mean(croppedImg, mask)[0];
        return score;
    }

    std::vector<std::vector<std::vector<int>>> PostProcessor::boxes_from_bitmap(
        const cv::Mat pred, const cv::Mat bitmap, const float& box_thresh,
        const float& det_db_unclip_ratio, const std::string& det_db_score_mode) {
        constexpr int max_candidates = 1000;
        int width = bitmap.cols;
        int height = bitmap.rows;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                         cv::CHAIN_APPROX_SIMPLE);
        int num_contours =
            contours.size() >= max_candidates ? max_candidates : contours.size();
        std::vector<std::vector<std::vector<int>>> boxes;
        for (int _i = 0; _i < num_contours; _i++) {
            constexpr int min_size = 3;
            if (contours[_i].size() <= 2) {
                continue;
            }
            float ssid;
            cv::RotatedRect box = cv::minAreaRect(contours[_i]);
            auto array = get_mini_boxes(box, ssid);
            auto box_for_unclip = array;
            // end get_mini_box
            if (ssid < min_size) {
                continue;
            }
            float score;
            if (det_db_score_mode == "slow") /* compute using polygon*/
                score = polygon_score_acc(contours[_i], pred);
            else
                score = box_score_fast(array, pred);

            if (score < box_thresh) continue;
            // start for unclip
            cv::RotatedRect points = un_clip(box_for_unclip, det_db_unclip_ratio);
            if (points.size.height < 1.001 && points.size.width < 1.001) {
                continue;
            }
            // end for unclip
            cv::RotatedRect clip_box = points;
            auto clip_array = get_mini_boxes(clip_box, ssid);

            if (ssid < min_size + 2) continue;

            int dest_width = pred.cols;
            int dest_height = pred.rows;
            std::vector<std::vector<int>> int_clip_array;
            for (int num_pt = 0; num_pt < 4; num_pt++) {
                std::vector<int> a{
                    static_cast<int>(clamp<float>(
                        roundf(clip_array[num_pt][0] / static_cast<float>(width) * static_cast<float>(dest_width)),
                        0, static_cast<float>(dest_width))),
                    static_cast<int>(clamp<float>(
                        roundf(clip_array[num_pt][1] / static_cast<float>(height) * static_cast<float>(dest_height)),
                        0, static_cast<float>(dest_height)))
                };
                int_clip_array.push_back(a);
            }
            boxes.push_back(int_clip_array);
        }
        return boxes;
    }

    std::vector<std::vector<std::vector<int>>> PostProcessor::filter_tag_det_res(
        std::vector<std::vector<std::vector<int>>> boxes,
        const std::array<int, 4>& det_img_info) {
        const int ori_img_w = det_img_info[0];
        const int ori_img_h = det_img_info[1];
        const float ratio_w = static_cast<float>(det_img_info[2]) / static_cast<float>(ori_img_w);
        const float ratio_h = static_cast<float>(det_img_info[3]) / static_cast<float>(ori_img_h);
        std::vector<std::vector<std::vector<int>>> root_points;
        for (int n = 0; n < boxes.size(); n++) {
            boxes[n] = order_points_clockwise(boxes[n]);
            for (int m = 0; m < boxes[0].size(); m++) {
                boxes[n][m][0] /= ratio_w;
                boxes[n][m][1] /= ratio_h;
                boxes[n][m][0] = _min(_max(boxes[n][m][0], 0), ori_img_w - 1);
                boxes[n][m][1] = _min(_max(boxes[n][m][1], 0), ori_img_h - 1);
            }
        }
        for (int n = 0; n < boxes.size(); n++) {
            const int rect_width = static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                pow(boxes[n][0][1] - boxes[n][1][1], 2)));
            const int rect_height = static_cast<int>(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                pow(boxes[n][0][1] - boxes[n][3][1], 2)));
            if (rect_width <= 4 || rect_height <= 4) continue;
            root_points.push_back(boxes[n]);
        }
        return root_points;
    }
}
