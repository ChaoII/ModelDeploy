#!/bin/bash
# 构建 tpu-mlir ONNX->bmodel 转换镜像 (Sophgo BM1688)
# 前置：将 tpu_mlir-1.27-py3-none-any.whl 与 tpu-mlir-resource.tar 放入本目录
#       (从算能官方 Sophon SDK 获取)
set -e
cd "$(dirname "$0")"
if [ ! -f tpu_mlir-1.27-py3-none-any.whl ]; then
    echo "缺少 tpu_mlir-1.27-py3-none-any.whl，请放入本目录后重试" >&2
    exit 1
fi
if [ ! -f tpu-mlir-resource.tar ]; then
    echo "缺少 tpu-mlir-resource.tar，请放入本目录后重试" >&2
    exit 1
fi
docker build -t tpuc_dev:1.27 .
echo "镜像构建完成: tpuc_dev:1.27"
