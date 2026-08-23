#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/tools/vad_segment.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT StreamingSTT : public SolutionBase {
public:
    explicit StreamingSTT(std::function<void(const std::string&)> on_text = nullptr);
    void push(const std::vector<float>& data, int sr);
    void set_on_text(std::function<void(const std::string&)> cb) { on_text_ = std::move(cb); }
    void run_once();
private:
    std::function<void(const std::string&)> on_text_;
    tool::VadSegment vad_;
    std::vector<float> pending_;
};
} // namespace modeldeploy::audio::solution
