#include "csrc/audio/speaker_verify/ecapa.h"
#include <algorithm>
#include <csrc/utils/utils.h>
#include <kaldi-native-fbank/csrc/feature-fbank.h>
#include <kaldi-native-fbank/csrc/online-feature.h>

namespace modeldeploy::audio::speaker_verify {
    SpeakerVerify::SpeakerVerify(const std::string& model_file,
                                 const RuntimeOption& custom_option) {
        runtime_option = custom_option;
        runtime_option.set_model_path(model_file);
        initialized_ = initialize();
    }

    std::unique_ptr<SpeakerVerify> SpeakerVerify::clone() const {
        auto clone_model = std::unique_ptr<SpeakerVerify>(new SpeakerVerify());
        clone_model->set_runtime(const_cast<SpeakerVerify*>(this)->clone_runtime());
        clone_model->runtime_option = runtime_option;
        clone_model->mel_bins_ = mel_bins_;
        clone_model->embedding_dim_ = embedding_dim_;
        clone_model->initialized_ = initialized_;
        return clone_model;
    }

    bool SpeakerVerify::is_initialized() const { return initialized_; }

    bool SpeakerVerify::initialize() {
        if (!init_runtime()) {
            MD_LOG_ERROR << "SpeakerVerify: failed to init runtime." << std::endl;
            return false;
        }
        // 探测 embedding 维度（输出张量最后一维）
        if (num_outputs() > 0) {
            auto out_info = get_output_info(0);
            const auto& shape = out_info.shape;
            if (!shape.empty()) embedding_dim_ = static_cast<int32_t>(shape.back());
        }
        if (embedding_dim_ <= 0) {
            MD_LOG_WARN << "SpeakerVerify: could not detect embedding dim, default 192." << std::endl;
            embedding_dim_ = 192;
        }
        return true;
    }

    bool SpeakerVerify::preprocess(const std::vector<float>& data, std::vector<Tensor>* outputs) {
        if (data.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: input data is empty." << std::endl;
            return false;
        }
        knf::FbankOptions opts;
        opts.frame_opts.dither = 0;
        opts.frame_opts.snip_edges = false;
        opts.frame_opts.window_type = "hamming";
        opts.frame_opts.samp_freq = 16000;
        opts.mel_opts.num_bins = mel_bins_;
        knf::OnlineFbank kaldi_f_bank(opts);
        kaldi_f_bank.AcceptWaveform(16000, data.data(), static_cast<int32_t>(data.size()));
        kaldi_f_bank.InputFinished();
        const int32_t n = kaldi_f_bank.NumFramesReady();
        std::vector<float> feats;
        feats.reserve(static_cast<size_t>(n) * mel_bins_);
        for (int32_t i = 0; i < n; ++i) {
            const auto* frame = kaldi_f_bank.GetFrame(i);
            for (int32_t k = 0; k < mel_bins_; ++k) feats.push_back(frame[k]);
        }
        if (feats.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: no fbank frames produced." << std::endl;
            return false;
        }
        // ECAPA 输入 [1, mel_bins, T]（以模型实际 shape 为准；若为 [1, T, mel_bins] 则交换）
        const int64_t T = static_cast<int64_t>(n);
        if (T == 0) return false;
        // 缺省列为 [1,80,T]，实现时按 get_input_info(0).shape 对齐（此处用标准 ECAPA 布局）
        const std::vector<int64_t> shape = {1, mel_bins_, T};
        std::vector<float> transposed(feats.size());
        // frame-major( T x mel ) -> channel-major( mel x T )
        for (int64_t t = 0; t < T; ++t)
            for (int32_t k = 0; k < mel_bins_; ++k)
                transposed[k * T + t] = feats[t * mel_bins_ + k];
        outputs->resize(1);
        (*outputs)[0] = std::move(Tensor(transposed.data(), shape, DataType::FP32, Device::CPU));
        return true;
    }

    bool SpeakerVerify::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* embedding) {
        if (infer_result.empty()) {
            MD_LOG_ERROR << "SpeakerVerify: no inference result." << std::endl;
            return false;
        }
        auto& t = infer_result[0];
        const auto* p = static_cast<const float*>(t.data());
        const int64_t stride = t.size();
        for (int64_t i = 0; i < stride; ++i) embedding->push_back(p[i]);
        // 模型输出通常已归一化；若需 L2 再归一化由应用/库层做，这里原样返回。
        return true;
    }

    bool SpeakerVerify::predict(const std::vector<float>& data, std::vector<float>* embedding) {
        if (!preprocess(data, &reused_input_tensors_)) return false;
        for (int i = 0; i < reused_input_tensors_.size(); i++)
            reused_input_tensors_[i].set_name(get_input_info(i).name);
        if (!infer(reused_input_tensors_, &reused_output_tensors_)) {
            MD_LOG_ERROR << "SpeakerVerify: inference failed." << std::endl;
            return false;
        }
        if (!postprocess(reused_output_tensors_, embedding)) return false;
        return true;
    }
} // namespace modeldeploy::audio::speaker_verify
