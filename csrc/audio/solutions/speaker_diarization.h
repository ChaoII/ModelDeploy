#pragma once
#include <vector>
#include <functional>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio { namespace speaker_verify { class SpeakerVerify; } }
namespace modeldeploy::audio::solution {
struct MODELDEPLOY_CXX_EXPORT Segment { int start_ms; int end_ms; int speaker_id; };
class MODELDEPLOY_CXX_EXPORT SpeakerDiarization : public SolutionBase {
public:
    // 说话人 embedding 回调：输入一个 VAD 切出的语音段（16k float PCM）与采样率，返回 embedding。
    using EmbedFn = std::function<std::vector<float>(const std::vector<float>&, int)>;

    std::vector<int> assign_speakers(const std::vector<std::vector<float>>& embeddings,
                                     float threshold = 0.7f) const;

    // VAD 分段 -> 逐段取 embedding -> 聚类出说话人也写入 Segment。
    // embed 为空时说话人标 -1（仅分段）。
    bool run(const std::vector<float>& audio, std::vector<Segment>* out,
             const EmbedFn& embed = nullptr, float threshold = 0.7f);

    // 把真实 SpeakerVerify(ECAPA) 包成 EmbedFn。
    static EmbedFn ecapa_embedder(speaker_verify::SpeakerVerify& model);
};
} // namespace modeldeploy::audio::solution
