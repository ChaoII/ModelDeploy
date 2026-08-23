#include "audio/solutions/speaker_diarization.h"
#include "audio/tools/vad_segment.h"
#include "vision/utils.h"

namespace modeldeploy::audio::solution {
std::vector<int> SpeakerDiarization::assign_speakers(
        const std::vector<std::vector<float>>& embeddings, float threshold) const {
    std::vector<int> ids(embeddings.size(), -1);
    std::vector<std::vector<float>> centroids;
    int next = 0;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto q = vision::utils::l2_normalize(embeddings[i]);
        int best = -1; float best_sim = threshold;
        for (size_t c = 0; c < centroids.size(); ++c) {
            const float s = vision::utils::compute_similarity(q, centroids[c]);
            if (s >= best_sim) { best_sim = s; best = (int)c; }
        }
        if (best < 0) {
            best = (int)centroids.size();
            centroids.push_back(q);
            ids[i] = best;
            if (best >= next) next = best + 1;
        } else {
            ids[i] = best;
            const float w = 0.5f;
            auto& c = centroids[(size_t)best];
            for (size_t k = 0; k < c.size() && k < q.size(); ++k) c[k] = c[k] * (1 - w) + q[k] * w;
        }
    }
    (void)next;
    return ids;
}

bool SpeakerDiarization::run(const std::vector<float>& audio, std::vector<Segment>* out) {
    if (!out) return false;
    tool::VadSegment vad(16000);
    vad.feed(audio);
    auto segs = vad.segments();
    out->clear();
    for (const auto& s : segs) out->push_back(Segment{s.start_ms, s.end_ms, -1});
    return true;
}
} // namespace modeldeploy::audio::solution
