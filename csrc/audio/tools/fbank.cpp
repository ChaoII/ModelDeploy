#include "audio/tools/fbank.h"
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>
namespace modeldeploy::audio::tool {
Fbank::Fbank(int sample_rate, int num_bins) : sr_(sample_rate), bins_(num_bins) {}
std::vector<std::vector<float>> Fbank::compute(const std::vector<float>& samples) const {
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.samp_freq = (float)sr_;
    opts.mel_opts.num_bins = bins_;
    knf::OnlineFbank kaldi_fbank(opts);
    if (!samples.empty())
        kaldi_fbank.AcceptWaveform((float)sr_, samples.data(), static_cast<int32_t>(samples.size()));
    kaldi_fbank.InputFinished();
    std::vector<std::vector<float>> out;
    const int n = kaldi_fbank.NumFramesReady();
    out.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        const auto* frame = kaldi_fbank.GetFrame(i);
        std::vector<float> row((size_t)bins_);
        for (int k = 0; k < bins_; ++k) row[(size_t)k] = frame[k];
        out.push_back(std::move(row));
    }
    return out;
}
} // namespace modeldeploy::audio::tool
