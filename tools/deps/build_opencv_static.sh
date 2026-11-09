#!/usr/bin/env bash
# 在 manylinux_2_28 容器（AlmaLinux 8 / glibc 2.28 / GCC 14）里构建 OpenCV 静态库。
# 用法: build_opencv_static.sh [OPENCV_TAG] [PKG_DIR_NAME]
#   OPENCV_TAG    默认 5.1.0（GitHub tag，脚本会下载源码 zip）
#   PKG_DIR_NAME  默认 opencv_5_x_linux_x64_static
# 产物: dist/<PKG_DIR_NAME>_glibc2.28.zip（布局与原 modelscope 包一致：<PKG>/{include,lib,...}）
#
# 说明：编译开关与 Windows 预编译包一致（关闭 dnn/calib/objdetect/stitching/ml/gapi，
# 只留 core/imgproc/imgcodecs/highgui/videoio/features/flann/geometry）；静态库；
# 在 glibc 2.28 用户空间内构建，产物可运行于 Ubuntu 20.04+/Debian 10+/RHEL 8+。
set -euo pipefail

TAG="${1:-5.1.0}"
PKG="${2:-opencv_5_x_linux_x64_static}"
# 安装布局：x86_64 包用 lib/，aarch64 包用 lib64/（与 cmake/opencv.cmake 的查找路径一致）
if [[ "$PKG" == *aarch64* ]]; then
  LIBDIR="lib64"
else
  LIBDIR="lib"
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-${ROOT}/.opencv-build}"
DIST="${DIST:-${ROOT}/dist}"

mkdir -p "$WORK" "$DIST"

# 工具链：容器自带 gcc-toolset，CMake 统一用 pip 的最新 3.31.x（仓库要求 >= 3.27）
export PATH="/opt/python/cp312-cp312/bin:${PATH}"
TOOLSET=$(ls -d /opt/rh/gcc-toolset-*/root/usr/bin 2>/dev/null | sort -V | tail -1 || true)
if [ -n "${TOOLSET:-}" ]; then
  export PATH="${TOOLSET}:${PATH}"
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  python -m pip install --no-cache-dir "cmake==3.31.6" ninja
fi
# 容器里可能自带旧版 cmake 但没有 ninja：统一用 pip 版覆盖，避免版本/生成器不一致
python -m pip install --no-cache-dir "cmake==3.31.6" ninja >/dev/null 2>&1 || true
hash -r
if ! command -v unzip >/dev/null 2>&1 || ! command -v zip >/dev/null 2>&1; then
  (dnf install -y unzip zip || yum install -y unzip zip)
fi

echo "== toolchain =="
gcc --version | sed -n 1p
ldd --version | sed -n 1p
cmake --version | sed -n 1p

SRC_ZIP="${WORK}/opencv-${TAG}.zip"
SRC_DIR="${WORK}/opencv-${TAG}"
BUILD_DIR="${WORK}/build-${TAG}-${PKG}"
INSTALL_DIR="${WORK}/install/${PKG}"

if [ ! -f "$SRC_ZIP" ]; then
  curl -fL --http1.1 --retry 3 -o "$SRC_ZIP" "https://github.com/opencv/opencv/archive/refs/tags/${TAG}.zip"
fi
if [ ! -d "$SRC_DIR" ]; then
  rm -rf "$WORK/opencv-${TAG}"
  unzip -q "$SRC_ZIP" -d "$WORK"
  # GitHub tag 包解压后目录名可能是 opencv-<tag>
  [ -d "$SRC_DIR" ] || mv "$WORK/opencv-${TAG#v}" "$SRC_DIR"
fi

# 编译开关与 options.txt 保持一致
OPTS=(
  -DBUILD_ZLIB=ON
  -DBUILD_TIFF=ON
  -DBUILD_OPENJPEG=OFF
  -DBUILD_JASPER=OFF
  -DBUILD_JPEG=ON
  -DBUILD_PNG=ON
  -DBUILD_OPENEXR=OFF
  -DBUILD_WEBP=OFF
  -DBUILD_TBB=OFF
  -DBUILD_IPP_IW=OFF
  -DBUILD_ITT=OFF
  -DWITH_AVFOUNDATION=OFF
  -DWITH_CAP_IOS=OFF
  -DWITH_CAROTENE=OFF
  -DWITH_CPUFEATURES=OFF
  -DWITH_EIGEN=OFF
  -DWITH_FFMPEG=OFF
  -DWITH_GSTREAMER=OFF
  -DWITH_GTK=OFF
  -DWITH_IPP=OFF
  -DWITH_HALIDE=OFF
  -DWITH_VULKAN=OFF
  -DWITH_INF_ENGINE=OFF
  -DWITH_NGRAPH=OFF
  -DWITH_JASPER=OFF
  -DWITH_OPENJPEG=OFF
  -DWITH_JPEG=ON
  -DWITH_WEBP=OFF
  -DWITH_OPENEXR=OFF
  -DWITH_PNG=ON
  -DWITH_TIFF=ON
  -DWITH_OPENVX=OFF
  -DWITH_GDCM=OFF
  -DWITH_TBB=OFF
  -DWITH_HPX=OFF
  -DWITH_OPENMP=OFF
  -DWITH_PTHREADS_PF=OFF
  -DWITH_V4L=OFF
  -DWITH_CLP=OFF
  -DWITH_OPENCL=OFF
  -DWITH_OPENCL_SVM=OFF
  -DWITH_VA=OFF
  -DWITH_VA_INTEL=OFF
  -DWITH_ITT=OFF
  -DWITH_PROTOBUF=OFF
  -DWITH_IMGCODEC_HDR=OFF
  -DWITH_IMGCODEC_SUNRASTER=OFF
  -DWITH_IMGCODEC_PXM=OFF
  -DWITH_IMGCODEC_PFM=OFF
  -DWITH_QUIRC=OFF
  -DWITH_ANDROID_MEDIANDK=OFF
  -DWITH_TENGINE=OFF
  -DWITH_ONNX=OFF
  -DWITH_TIMVX=OFF
  -DWITH_OBSENSOR=OFF
  -DWITH_CANN=OFF
  -DWITH_FLATBUFFERS=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_opencv_apps=OFF
  -DBUILD_ANDROID_PROJECTS=OFF
  -DBUILD_ANDROID_EXAMPLES=OFF
  -DBUILD_DOCS=OFF
  -DBUILD_EXAMPLES=OFF
  -DBUILD_PACKAGE=OFF
  -DBUILD_PERF_TESTS=OFF
  -DBUILD_TESTS=OFF
  -DBUILD_WITH_STATIC_CRT=OFF
  -DBUILD_FAT_JAVA_LIB=OFF
  -DBUILD_ANDROID_SERVICE=OFF
  -DBUILD_JAVA=OFF
  -DBUILD_OBJC=OFF
  -DBUILD_KOTLIN_EXTENSIONS=OFF
  -DENABLE_PRECOMPILED_HEADERS=OFF
  -DENABLE_FAST_MATH=ON
  -DCV_TRACE=OFF
  -DBUILD_opencv_java=OFF
  -DBUILD_opencv_gapi=OFF
  -DBUILD_opencv_objc=OFF
  -DBUILD_opencv_js=OFF
  -DBUILD_opencv_ts=OFF
  -DBUILD_opencv_python2=OFF
  -DBUILD_opencv_python3=OFF
  -DBUILD_opencv_dnn=OFF
  -DBUILD_opencv_calib=OFF
  -DBUILD_opencv_objdetect=OFF
  -DBUILD_opencv_stitching=OFF
  -DBUILD_opencv_ml=OFF
  -DBUILD_opencv_flann=ON
  -DBUILD_opencv_highgui=ON
  -DBUILD_opencv_geometry=ON
  -DBUILD_opencv_imgproc=ON
  -DBUILD_opencv_imgcodecs=ON
  -DBUILD_opencv_videoio=ON
  -DBUILD_opencv_features=ON
)

rm -rf "$BUILD_DIR" "$INSTALL_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_INSTALL_LIBDIR="$LIBDIR" \
  -DOPENCV_GENERATE_PKGCONFIG=OFF \
  "${OPTS[@]}"
cmake --build "$BUILD_DIR" -j "$(nproc)"
cmake --install "$BUILD_DIR"

echo "== installed layout =="
ls -la "$INSTALL_DIR"
ls -la "$INSTALL_DIR/$LIBDIR/cmake/opencv5"

OUT="${DIST}/${PKG}_glibc2.28.zip"
rm -f "$OUT"
(cd "${WORK}/install" && zip -qr "$OUT" "$PKG")
echo "== artifact =="
ls -lh "$OUT"
sha256sum "$OUT"
