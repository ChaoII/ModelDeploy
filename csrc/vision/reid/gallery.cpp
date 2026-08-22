//
// Created for in-memory pedestrian Re-ID feature gallery.
//

#include "vision/reid/gallery.h"
#include "vision/utils.h"

#include <algorithm>

namespace modeldeploy::vision::reid {
    void ReIdGallery::enroll(const std::string& label, const std::vector<float>& embedding) {
        gallery_[label] = embedding;   // 覆盖同 label
    }

    std::vector<bool> ReIdGallery::remove(const std::string& label) {
        return {gallery_.erase(label) > 0};
    }

    std::vector<std::pair<std::string, float>> ReIdGallery::match(
            const std::vector<float>& embedding, int k) const {
        std::vector<std::pair<float, std::string>> scored;
        scored.reserve(gallery_.size());
        for (const auto& [label, feat] : gallery_) {
            scored.emplace_back(utils::compute_similarity(feat, embedding), label);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (k > 0 && static_cast<size_t>(k) < scored.size()) {
            scored.resize(k);
        }
        std::vector<std::pair<std::string, float>> out;
        out.reserve(scored.size());
        for (auto& [s, label] : scored) {
            out.emplace_back(std::move(label), s);
        }
        return out;
    }
} // namespace modeldeploy::vision::reid
