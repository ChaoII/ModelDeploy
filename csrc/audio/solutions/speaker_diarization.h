#pragma once
#include <vector>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
struct MODELDEPLOY_CXX_EXPORT Segment { int start_ms; int end_ms; int speaker_id; };
class MODELDEPLOY_CXX_EXPORT SpeakerDiarization : public SolutionBase {
public:
    std::vector<int> assign_speakers(const std::vector<std::vector<float>>& embeddings,
                                     float threshold = 0.7f) const;
    bool run(const std::vector<float>& audio, std::vector<Segment>* out);
};
} // namespace modeldeploy::audio::solution
