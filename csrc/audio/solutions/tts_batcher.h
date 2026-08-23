#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT TTSBatcher : public SolutionBase {
public:
    explicit TTSBatcher(std::function<std::vector<float>(const std::string&)> synth = nullptr);
    void enqueue(const std::vector<std::string>& texts);
    std::vector<std::vector<float>> dequeue_all();
    size_t pending() const { return queue_.size(); }
private:
    std::function<std::vector<float>(const std::string&)> synth_;
    std::vector<std::string> queue_;
};
} // namespace modeldeploy::audio::solution
