# Sophgo tpu-mlir ONNX -> bmodel 转换镜像（BM1688 / CV186AH）
# 国内源优化：apt 用清华源、pip 用清华源、基础镜像走国内 registry 代理。
# 注意：tpu-mlir 仅支持 Python 3.10（Requires-Python: >=3.10,<3.11），故基镜像用 python:3.10-slim。
# 前置：将 tpu_mlir-1.14-py3-none-any.whl 与 tpu-mlir-resource.tar 放入本目录
#       （从算能官方 Sophon SDK 获取）。
# 构建：./build_docker.sh
FROM dockerproxy.net/library/python:3.10-slim

# apt 换清华源（debian bookworm）
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g; s@security.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g; s@security.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list || true
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip 换清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

WORKDIR /opt

# 拷贝 tpu-mlir 离线包（避免镜像内联网下载 200MB）
COPY tpu_mlir-1.14-py3-none-any.whl /opt/
COPY tpu-mlir-resource.tar /opt/
RUN tar -xf /opt/tpu-mlir-resource.tar -C /opt && rm -f /opt/tpu-mlir-resource.tar

# 先装 tpu-mlir（--no-deps，避免依赖冲突），再装转换所需依赖（清华源）。
# onnxsim 无 aarch64 预编译 wheel，需源码编译且非转换必需，故不装；需要简化时在容器外做。
RUN pip install --no-cache-dir --no-deps /opt/tpu_mlir-1.14-py3-none-any.whl && \
    pip install --no-cache-dir \
        numpy==1.24.3 scipy==1.11.1 tqdm Pillow protobuf==3.20.3 \
        onnx onnxruntime pycocotools scikit-image graphviz && \
    rm -f /opt/tpu_mlir-1.14-py3-none-any.whl

ENV PATH="/usr/local/bin:/opt/llvm/bin:${PATH}"
