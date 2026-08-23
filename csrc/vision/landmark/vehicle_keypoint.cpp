#include "vision/landmark/vehicle_keypoint.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision::landmark {
    VehicleKeypoint::VehicleKeypoint(const std::string& model_file, const RuntimeOption& option)
        : pose_(model_file, option) {
        // 默认 4 车轮关键点；不同车型模型可用 set_keypoints_num 覆盖（spec §12 已强调点数泛化）。
        pose_.get_postprocessor().set_keypoints_num(4);
    }

    bool VehicleKeypoint::predict(const ImageData& img,
                                  std::vector<KeyPointsResult>* results,
                                  TimerArray* timer) {
        return pose_.predict(img, results, timer);
    }

    bool VehicleKeypoint::batch_predict(const std::vector<ImageData>& imgs,
                                        std::vector<std::vector<KeyPointsResult>>* results,
                                        TimerArray* timer) {
        return pose_.batch_predict(imgs, results, timer);
    }

    bool VehicleKeypoint::draw_result(ImageData& img,
                                      const std::vector<KeyPointsResult>& results,
                                      double threshold) {
        (void)threshold;
        img = vis_keypoints(img, results, "", 14, 4, 0.15, false);
        return true;
    }

    std::unique_ptr<VehicleKeypoint> VehicleKeypoint::clone() const {
        auto ret = std::make_unique<VehicleKeypoint>(*this);
        ret->pose_.set_runtime(ret->pose_.clone_runtime());
        return ret;
    }
} // namespace modeldeploy::vision::landmark
