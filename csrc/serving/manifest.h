#pragma once

#include <string>
#include <vector>
#include "core/md_decl.h"
#include "serving/model_repo.h"

namespace modeldeploy::serving {

struct ManifestModel {
    std::string id, display, type;
    std::string model_f, rec_f, cls_f, dict_f;
    std::vector<int> input_size;
    std::vector<std::string> labels;
    std::string desc;
};

MODELDEPLOY_CXX_EXPORT
bool load_manifest(const std::string& path, const std::string& base,
                   std::vector<ManifestModel>* out, std::string* err);

}  // namespace modeldeploy::serving
