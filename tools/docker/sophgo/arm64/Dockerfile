# Sophgo BM1688/CV186X aarch64 交叉编译环境（Docker + QEMU arm64 模拟）
#
# 用途：在 x86 主机上用 QEMU 模拟 arm64 的 Ubuntu 22.04，配合算能官方 aarch64 预编译
# libsophon（bmrt/bmcv/bmlib，SophonSDK 1.9 / 0.4.11）编译 ModelDeploy 的 Sophgo 后端，
# 产出原生 aarch64 二进制直接拷贝到 BM1688 板子运行。
#
# 本镜像只含 libsophon 与编译工具链，不含 OpenCV——OpenCV 由用户提供 aarch64 静态库，
# 编译时通过 -DOpenCV_DIR 指向。版本必须与目标板子匹配
# （bm1688: ls /opt/sophon，libsophon-0.4.11）。
#
# 前置：libsophon-clean.tar.gz 需先用 amd64 容器重新打包（见 build_sdk_tars.sh），
#   因为 QEMU arm64 下 GNU tar 解包报 "Invalid argument"，必须用 python3 tarfile 解压。
#   docker build --platform linux/arm64 -t tpuc_dev:arm64-sophgo .
#
# 编译用法（OpenCV 静态库目录如 <opencv>/include、<opencv>/lib 挂到 /opencv）：
#   docker run --rm --platform linux/arm64 \
#       -v <源码根目录>:/workspace -v <构建目录>:/build -v <opencv安装目录>:/opencv \
#       -w /workspace tpuc_dev:arm64-sophgo bash -c "
#         cmake -S /workspace -B /build -G Ninja \
#           -DCMAKE_BUILD_TYPE=Release \
#           -DENABLE_SOPHGO=ON -DENABLE_ORT=OFF -DENABLE_MNN=OFF -DENABLE_TRT=OFF \
#           -DWITH_GPU=OFF -DBUILD_AUDIO=OFF -DBUILD_VISION=ON \
#           -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=ON \
#           -DBUILD_ENCRYPTION=OFF \
#           -DOpenCV_DIR=/opencv/lib/cmake/opencv4 && \
#         cmake --build /build --target demo_detection_sophgo -j \$(nproc)"
#   产物 /build/bin/demo_detection_sophgo 为 aarch64 原生二进制，拷到板子运行。
#
# 注意：若用户 OpenCV 头文件(mat.hpp)也 include bmcv_api_ext.h，可能与项目
#   basic_types.h 的 ROTATE_* 枚举冲突，此时需改项目 RotateFlags 枚举名。

FROM dockerproxy.net/library/ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build pkg-config \
    gcc g++ g++-aarch64-linux-gnu \
    libgomp1 python3 python3-pip wget ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# 升级 CMake：Ubuntu 22.04 自带 3.22 的 FetchContent+DOWNLOAD_EXTRACT_TIMESTAMP
# 有解析 bug（URL_HASH 与后续参数被合并），需 >=3.27
RUN pip3 install --no-cache-dir cmake && \
    cmake --version | head -1

# 算能 aarch64 预编译 libsophon（clean 重打包版，QEMU 下用 python3 tarfile 解压）
COPY libsophon-clean.tar.gz /opt/
RUN python3 -c "import tarfile; tarfile.open('/opt/libsophon-clean.tar.gz').extractall('/opt')" && \
    rm -f /opt/*.tar.gz && \
    mv /opt/libsophon_soc_0.4.11_aarch64/opt/sophon /opt/sophon && \
    rm -rf /opt/libsophon_soc_0.4.11_aarch64

# libsophon-current 符号链接（sophgo.cmake 自动探测路径）
RUN ln -sf /opt/sophon/libsophon-0.4.11 /opt/sophon/libsophon-current

# 运行时库路径（OpenCV 用户自行挂载）
ENV LD_LIBRARY_PATH="/opt/sophon/libsophon-current/lib"
ENV SOPHGO_SDK_DIR=/opt/sophon/libsophon-current

WORKDIR /workspace
