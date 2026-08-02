#pragma once

// SCRFD 预处理：
// - CPU 端已由 VisionProcessorBackend::scrfd_preprocess 的 fused SIMD 通道取代
//   （见 processors/cpu/cpu_processor_backend.cpp），不再需要独立标量 kernel。
// - CUDA 端见 scrfd_preproc.cuh / scrfd_preproc.cu。

#include "core/tensor.h"
#include "vision/common/image_data.h"
