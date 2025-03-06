// sherpa-onnx/csrc/online-paraformer-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_DECODER_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OnlineParaformerDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;

  int32_t last_non_blank_frame_index = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_PARAFORMER_DECODER_H_
