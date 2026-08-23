#include "audio/tools/vad_segment.h"
#include <algorithm>
#include <cmath>
namespace modeldeploy::audio::tool {
std::vector<Seg> VadSegment::segments() const {
    std::vector<Seg> out;
    const int win = sr_ / 100;
    const size_t nwin = buf_.size() / (size_t)win;
    if (nwin == 0) return out;
    std::vector<bool> speech(nwin, false);
    for (size_t w = 0; w < nwin; ++w) {
        float e = 0.0f;
        for (int i = 0; i < win; ++i) { const float v = buf_[w*(size_t)win + i]; e += v * v; }
        speech[w] = (std::sqrt(e / win) > thr_);
    }
    int start = -1;
    std::vector<int> on;
    for (size_t w = 0; w <= nwin; ++w) {
        const bool sp = (w < nwin) ? speech[w] : false;
        if (sp && start < 0) start = (int)w;
        else if (!sp && start >= 0) {
            on.push_back(start); on.push_back((int)w - 1);
            start = -1;
        }
    }
    std::vector<Seg> merged;
    int cur_start = -1, cur_end = -1;
    auto flush = [&](){ if (cur_start >= 0) {
        const int start_ms = cur_start * 10, end_ms = (cur_end + 1) * 10;
        const size_t b = (size_t)cur_start * win, e = (size_t)(cur_end + 1) * win;
        if ((end_ms - start_ms) >= min_speech_) {
            Seg s; s.start_ms = start_ms; s.end_ms = end_ms;
            s.samples.assign(buf_.begin() + (long)b, buf_.begin() + (long)std::min(e, buf_.size()));
            merged.push_back(std::move(s));
        }
    } cur_start = cur_end = -1; };
    for (size_t i = 0; i < on.size(); i += 2) {
        const int s = on[i], e = on[i+1];
        if (cur_start < 0) { cur_start = s; cur_end = e; }
        else if ((s - cur_end - 1) * 10 < min_silence_) { cur_end = e; }
        else { flush(); cur_start = s; cur_end = e; }
    }
    flush();
    return merged;
}
} // namespace modeldeploy::audio::tool
