#include "csrc/audio/speaker_gallery.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::audio {
    void SpeakerGallery::enroll(const std::string& label, const std::vector<float>& embedding) {
        gallery_[label] = vision::utils::l2_normalize(embedding);   // 覆盖同 label
    }

    std::vector<bool> SpeakerGallery::remove(const std::string& label) {
        return {gallery_.erase(label) > 0};
    }

    std::vector<std::pair<std::string, float>>
    SpeakerGallery::match(const std::vector<float>& embedding, int k) const {
        const auto q = vision::utils::l2_normalize(embedding);
        std::vector<std::pair<std::string, float>> scored;
        scored.reserve(gallery_.size());
        for (const auto& [label, ref] : gallery_) {
            scored.emplace_back(label, vision::utils::compute_similarity(q, ref));
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        if (k > 0 && static_cast<size_t>(k) < scored.size()) scored.resize(k);
        return scored;
    }
} // namespace modeldeploy::audio
