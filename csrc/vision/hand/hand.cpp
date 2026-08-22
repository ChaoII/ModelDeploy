#include "vision/hand/hand.h"
#include "vision/common/visualize/visualize.h"

namespace modeldeploy::vision::hand {
    HandKeypoint::HandKeypoint(const std::string& model_file, const RuntimeOption& option)
        : pose_(model_file, option) {
        pose_.get_postprocessor().set_keypoints_num(21);
    }

    bool HandKeypoint::predict(const ImageData& img,
                               std::vector<KeyPointsResult>* results,
                               TimerArray* timer) {
        return pose_.predict(img, results, timer);
    }

    bool HandKeypoint::batch_predict(const std::vector<ImageData>& imgs,
                                     std::vector<std::vector<KeyPointsResult>>* results,
                                     TimerArray* timer) {
        return pose_.batch_predict(imgs, results, timer);
    }

    bool HandKeypoint::draw_result(ImageData& img,
                                   const std::vector<KeyPointsResult>& results,
                                   double threshold) {
        (void)threshold;
        img = vis_hand(img, results, "", 14, 4, 0.15, false);
        return true;
    }

    std::unique_ptr<HandKeypoint> HandKeypoint::clone() const {
        auto ret = std::make_unique<HandKeypoint>(*this);
        ret->pose_.set_runtime(ret->pose_.clone_runtime());
        return ret;
    }
} // namespace modeldeploy::vision::hand
