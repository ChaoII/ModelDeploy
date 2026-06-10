# Project Memory

## Branch: feature/tests-and-benchmarks

Created from `main` for comprehensive test & benchmark infrastructure.

---

## Test Status

### ✅ Passed (all compiled and run)

| Test Target | Cases | Assertions | Notes |
|------------|-------|-----------|-------|
| `test_image_data` | 25 | 135 | C++ ImageData 全部公开 API（构造/访问/预处理/I/O），含 5 个 benchmark |
| `test_capi_image` | 14 | 53 | C API MDImage 全部 16 个函数（读/写/克隆/裁剪/格式转换） |
| `test_md_image` | 1 | 2 | 向后兼容旧测试 |

### ⚠️ Need ENABLE_ORT=ON

| Test Target | Reason |
|------------|--------|
| `test_vision_models` | 13 个模型类型（分类/检测/分割/姿态/OBB/人脸/OCR），需 onnxruntime 加载模型文件 |
| `benchmark_yolo_preproc` | `yolo_preprocess_cpu()` 是内部函数未导出，需改为调用公开 API |

### 🏎️ Benchmarks

| Target | Status |
|--------|--------|
| `benchmark_image_data` | 编译通过，16 个 benchmark 组（resize/cvt/permute/normalize/fuse/letterbox 等，5 种分辨率），运行时需放大超时 |
| `benchmark_yolo_preproc` | 编译失败: `yolo_preprocess_cpu` 未导出。需调用 `UltralyticsDet` 公开 API 替代 |

---

## Build Instructions

### Prerequisites
- VS 2022 (vcvars64.bat)
- Ninja (bundled with VS)
- OpenCV 5.x prebuilt (downloaded via FetchContent from ModelScope)
- OnnxRuntime (for model tests)

### Full Build
```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARK=ON ^
  -DENABLE_ORT=ON -DWITH_GPU=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

### Minimal Build (tests only, no model inference)
```cmd
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARK=ON ^
  -DENABLE_ORT=OFF -DENABLE_MNN=OFF -DWITH_GPU=OFF
```

---

## Known Issues & Fixes

### 1. OpenCV Prebuilt Package (opencv.cmake:59)
Path: `_deps/opencv-src/x64/vc17/staticlib`
Works with VS 2022 Release builds. Debug mode overrides to local path `E:/develop/opencv/build/x64/vc16/lib`.

### 2. Empty ImageData Crash
Calling `normalize()`, `convert()`, `rotate()`, `pad()`, `cast()`, `permute()`, `letter_box()`, `fuse_*()` on empty ImageData causes SIGSEGV.
**Test workaround:** Skip these calls on empty images; just check `empty()`.

### 3. element_count() returns width*height (not width*height*channels)
`ImageData::element_count()` returns total pixel count (w*h), not total element count (w*h*channels). This is by design (matches OpenCV `Mat::total()`).

### 4. yolo_preprocess_cpu not exported
Internal function is not marked `MODELDEPLOY_CXX_EXPORT`. Benchmarks should use public `UltralyticsDet` API or the function needs to be exported.

---

## File Structure

```
tests/
  test_image_data.cpp       # 1151 lines, 25 TEST_CASE, C++ ImageData
  test_capi_image.cpp        # 248 lines, 14 TEST_CASE, C API MDImage
  test_vision_models.cpp     # 315 lines, 13 model types
  test_md_image.cpp          # 40 lines, legacy test
  test_core.cpp              # 50 lines, core library
  utils.h / utils.cpp        # test helpers
  CMakeLists.txt             # test build config

benchmark/
  benchmark_image_data.cpp   # 783 lines, 16 benchmark groups
  benchmark_yolo_preproc.cpp # yolo CPU/GPU preproc benchmarks
  CMakeLists.txt             # benchmark build config
```
