#include "csrc/audio/speaker_gallery.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::audio {
    void SpeakerGallery::enroll(const std::string& label, const std::vector<float>& embedding) {
        gallery_[label] = vision::utils::l2_normalize(embedding);   // 覆盖同 label
    }

    std::vector<bool> SpeakerGallery::remove(const std::string& label) {
        std::vector<bool> ok{ false };
        auto it = gallery_.find(label);
        if (it != gallery_.end()) {
            gallery_.erase(it);
            ok[0] = true;
        }
        return ok;
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
        if (scored.size() > static_cast<size_t>(k)) scored.resize(k);
        return scored;
    }
} // namespace modeldeploy::audio
