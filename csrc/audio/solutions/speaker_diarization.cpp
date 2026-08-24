#include "audio/solutions/speaker_diarization.h"
#include "audio/tools/vad_segment.h"
#include "audio/speaker_verify/ecapa.h"
#include "vision/utils.h"

namespace modeldeploy::audio::solution {
std::vector<int> SpeakerDiarization::assign_speakers(
        const std::vector<std::vector<float>>& embeddings, float threshold) const {
    std::vector<int> ids(embeddings.size(), -1);
    std::vector<std::vector<float>> centroids;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto q = vision::utils::l2_normalize(embeddings[i]);
        if (q.empty()) { ids[i] = -1; continue; }
        int best = -1; float best_sim = threshold;
        for (size_t c = 0; c < centroids.size(); ++c) {
            const float s = vision::utils::compute_similarity(q, centroids[c]);
            if (s >= best_sim) { best_sim = s; best = (int)c; }
        }
        if (best < 0) {
            best = (int)centroids.size();
            centroids.push_back(q);
            ids[i] = best;
        } else {
            ids[i] = best;
            const float w = 0.5f;
            auto& c = centroids[(size_t)best];
            for (size_t k = 0; k < c.size() && k < q.size(); ++k) c[k] = c[k] * (1 - w) + q[k] * w;
        }
    }
    return ids;
}

bool SpeakerDiarization::run(const std::vector<float>& audio, std::vector<Segment>* out,
                             const EmbedFn& embed, float threshold) {
    if (!out) return false;
    tool::VadSegment vad(16000);
    vad.feed(audio);
    auto segs = vad.segments();
    out->clear();

    std::vector<int> ids;
    if (embed) {
        // 先逐段取 embedding；空 embedding 视为该段无法聚类（-1）
        std::vector<std::vector<float>> embs;
        embs.reserve(segs.size());
        bool any_embed = false;
        for (const auto& s : segs) {
            auto e = embed(s.samples, 16000);
            if (e.empty()) { embs.emplace_back(); }
            else { embs.push_back(std::move(e)); any_embed = true; }
        }
        if (any_embed) ids = assign_speakers(embs, threshold);
    }

    for (size_t i = 0; i < segs.size(); ++i) {
        Segment s;
        s.start_ms = segs[i].start_ms;
        s.end_ms = segs[i].end_ms;
        s.speaker_id = (ids.size() == segs.size()) ? ids[i] : -1;
        out->push_back(s);
    }
    return true;
}

SpeakerDiarization::EmbedFn SpeakerDiarization::ecapa_embedder(speaker_verify::SpeakerVerify& model) {
    return [&model](const std::vector<float>& data, int /*sr*/) {
        std::vector<float> emb;
        if (model.is_initialized() && model.predict(data, &emb)) return emb;
        return std::vector<float>{};
    };
}
} // namespace modeldeploy::audio::solution
