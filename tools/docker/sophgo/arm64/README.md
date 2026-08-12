# Sophgo BM1688 aarch64 交叉编译镜像

在 x86 主机上用 Docker + QEMU 模拟 arm64 的 Ubuntu 22.04，配合算能官方 aarch64 预编译
libsophon，交叉编译 ModelDeploy 的 Sophgo 后端，产出原生 aarch64 二进制直接拷贝到 BM1688 板子运行。

## 镜像

| 镜像 | libsophon | sophon-media | cmake | 说明 |
|------|-----------|--------------|-------|------|
| `tpuc_dev:arm64-sophgo-0414` | 0.4.14 | 2.2.0 | ≥3.27 (pip 安装) | SDK 2.2，推荐 |
| `tpuc_dev:arm64-sophgo-0414-opencv5` | 0.4.14 | 2.2.0 | ≥3.27 | 0414 + OpenCV5 gcc7.3 兼容符号链接 |
| `tpuc_dev:arm64-sophgo-0411-cmake-path` | 0.4.11 | 1.9.0 | ≥3.27 | SDK 1.9（旧版） |
| `tpuc_dev:arm64-sophgo-0411` | 0.4.11 | 1.9.0 | ≥3.27 | SDK 1.9 基础 |
| `tpuc_dev:arm64-sophgo` | 0.4.11 | — | ≥3.27 | SDK 1.9 精简 |

## 构建 0414 镜像

```bash
# 前置：从 Sophon SDK 2.2（sophonsdk_edge_v2.2_official_release）准备 clean tar：
#   libsophon_soc_0.4.14_aarch64.tar.gz  -> libsophon-clean-0.4.14.tar.gz
#   sophon-media-soc_2.2.0_aarch64.tar.gz -> sophon-media-clean-2.2.0.tar.gz
# 大二进制包不入 git，需手动从 SDK 生成（见下方"生成 clean tar"）。

docker build --platform linux/arm64 -f Dockerfile.0414 \
  -t tpuc_dev:arm64-sophgo-0414 .
# OpenCV5 版：
docker build --platform linux/arm64 -f Dockerfile.0414.opencv5 \
  -t tpuc_dev:arm64-sophgo-0414-opencv5 .
```

> 注意：QEMU arm64 下 GNU tar 解包报 `Invalid argument`，必须用 python3 tarfile 解压（Dockerfile 已处理）。
> 因此 `*-clean-*.tar.gz` 需在 amd64 容器中用 GNU tar 重打包（保留符号链接）。

## 生成 clean tar（从官方 SDK）

在 **amd64 Linux 容器**里重打包（Windows tar 无法保留 `.so` 符号链接，会损坏）：

```bash
# 准备官方 SDK 2.2 的两个 tar 到 /work，然后：
docker run --rm -v "$PWD:/work" ubuntu:22.04 bash -c '
  cd /work
  mkdir -p l && tar -xzf libsophon_soc_0.4.14_aarch64.tar.gz -C l \
    && tar -czf libsophon-clean-0.4.14.tar.gz -C l .
  mkdir -p m && tar -xzf sophon-media-soc_2.2.0_aarch64.tar.gz -C m \
    && tar -czf sophon-media-clean-2.2.0.tar.gz -C m .
'
```

## 交叉编译 ModelDeploy

```bash
docker run --rm --platform linux/arm64 \
  -v <源码根目录>:/workspace -w /workspace \
  tpuc_dev:arm64-sophgo-0414-opencv5 bash -c "
    cmake -S /workspace -B /workspace/build_sophgo -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SOPHGO=ON -DENABLE_ORT=OFF -DENABLE_MNN=OFF -DENABLE_TRT=OFF \
      -DWITH_GPU=OFF -DBUILD_AUDIO=OFF -DBUILD_VISION=ON \
      -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
      -DBUILD_ENCRYPTION=OFF && \
    cmake --build /workspace/build_sophgo -j \$(nproc)"
# 产物 /workspace/build_sophgo/bin/demo_detection_sophgo 为 aarch64 原生二进制
```

OpenCV 由 `cmake/opencv.cmake` 自动从 modelscope 下载 aarch64 静态库（见 Dockerfile.0414.opencv5 说明）。

## 注意

- modelscope OpenCV5 静态包由 gcc 7.3 构建，硬编码旧编译器路径，`Dockerfile.0414.opencv5` 已建兼容符号链接。
- 若 OpenCV 头文件也 include `bmcv_api_ext.h`，可能与项目 `basic_types.h` 的 ROTATE_* 枚举冲突，需改名。
