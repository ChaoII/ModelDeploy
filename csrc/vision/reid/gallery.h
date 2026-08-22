//
// Created for in-memory pedestrian Re-ID feature gallery.
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/md_decl.h"

namespace modeldeploy::vision::reid {
    /*! @brief In-memory Re-ID feature gallery: label -> embedding, top-k cosine match.
     *  匹配基于 utils::compute_similarity（对 L2 归一化向量做点积），
     *  因此 enroll/match 前 embedding 必须已 L2 归一化（ReID::predict 输出即如此）。
     */
    class MODELDEPLOY_CXX_EXPORT ReIdGallery {
    public:
        void clear() { gallery_.clear(); }

        /// 注册 label -> embedding，同 label 覆盖。
        void enroll(const std::string& label, const std::vector<float>& embedding);

        /// 移除 label，返回是否实际移除（vector<bool> 保持接口约定）。
        std::vector<bool> remove(const std::string& label);

        /// 与库内各条目做余弦相似度，降序返回 top-k (label, score)。
        std::vector<std::pair<std::string, float>> match(const std::vector<float>& embedding, int k) const;

        size_t size() const { return gallery_.size(); }

    private:
        std::map<std::string, std::vector<float>> gallery_;
    };
} // namespace modeldeploy::vision::reid
