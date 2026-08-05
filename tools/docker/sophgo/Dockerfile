FROM python:3.10-slim
RUN apt-get update && apt-get install -y gcc g++ wget libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
COPY tpu_mlir-1.27-py3-none-any.whl /opt/
COPY tpu-mlir-resource.tar /opt/
RUN tar -xf /opt/tpu-mlir-resource.tar -C /opt && rm -f /opt/tpu-mlir-resource.tar
RUN pip install --no-cache-dir /opt/tpu_mlir-1.27-py3-none-any.whl torch==2.1.0 onnx==1.14.1 onnxruntime==1.16.3 onnxsim==0.4.17
ENV PATH="/usr/local/bin:/opt/llvm/bin:${PATH}"
