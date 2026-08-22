//
// Created for in-memory speaker feature gallery.
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/md_decl.h"
#include "vision/utils.h"

namespace modeldeploy::audio {
    /*! @brief In-memory speaker gallery: label -> l2-normalized embedding, top-k cosine match.
     *  match 对 query 先 l2_normalize 再 compute_similarity（=余弦）。
     */
    class MODELDEPLOY_CXX_EXPORT SpeakerGallery {
    public:
        void clear() { gallery_.clear(); }
        void enroll(const std::string& label, const std::vector<float>& embedding);
        std::vector<bool> remove(const std::string& label);
        std::vector<std::pair<std::string, float>> match(const std::vector<float>& embedding, int k) const;
        size_t size() const { return gallery_.size(); }
    private:
        std::map<std::string, std::vector<float>> gallery_;
    };
} // namespace modeldeploy::audio
