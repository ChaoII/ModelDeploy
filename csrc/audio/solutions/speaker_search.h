#pragma once
#include <string>
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "audio/speaker_gallery.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT SpeakerSearch : public SolutionBase {
public:
    void enroll(const std::string& label, const std::vector<float>& embedding) { gallery_.enroll(label, embedding); }
    std::vector<std::pair<std::string,float>> match(const std::vector<float>& embedding, int k = 1) const { return gallery_.match(embedding, k); }
    size_t size() const { return gallery_.size(); }
private:
    audio::SpeakerGallery gallery_;
};
} // namespace modeldeploy::audio::solution
