cmake_minimum_required(VERSION 3.27)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 17)

set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE "")
set(DEPENDS "")
set(PRIVATE_DEPENDS "")

message("==================${CMAKE_BUILD_TYPE}===================")
option(ENABLE_ORT "enable ort backend" ON)
option(ENABLE_MNN "enable mnn backend" ON)
option(ENABLE_TRT "enable trt backend" OFF)
option(ENABLE_SOPHGO "enable sophgo backend (Linux + SOPHON-Sail)" OFF)
option(ENABLE_VAAPI "enable VAAPI hardware video path (Linux + libva; compile-gated, only affects video backends)" OFF)
option(ENABLE_WETEXT "enable WeTextProcessing ITN backend (requires OpenFst + we-text-processing)" OFF)
option(BUILD_AUDIO "build asr tts vad sr" ON)
option(BUILD_NLP "build nlp module (jieba tokenizer + text tools)" OFF)
option(BUILD_VISION "build vision module" ON)
option(BUILD_BARCODE  "Enable barcode/QR recognition (ZXing-C++)" ON)
option(BUILD_CAPI "build with capi" ON)
option(BUILD_PYTHON "build with python" ON)
option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build tests" OFF)
option(BUILD_BENCHMARK "build benchmarks" OFF)
option(BUILD_SURVEILLANCE "build AI surveillance application" OFF)
option(BUILD_VIDEO "build video decoder module (requires FFmpeg and BUILD_VISION)" OFF)
option(ENABLE_FFMPEG "build video FFmpeg backend" ON)
option(ENABLE_GSTREAMER "build video GStreamer backend" OFF)
option(BUILD_ENCRYPTION "enable model encryption/decryption (requires OpenSSL)" ON)
option(WITH_GPU "Enable GPU" ON)
option(BUILD_ON_JETSON "Whether to build ModelDeploy on Nvidia Jetson" OFF)

if (BUILD_SURVEILLANCE)
    set(BUILD_VIDEO ON CACHE BOOL "build video decoder module (required by BUILD_SURVEILLANCE)" FORCE)
endif ()

if (NOT BUILD_ON_JETSON)
    if (EXISTS "/etc/nv_tegra_release")
        message(STATUS "detect Jetson")
        set(BUILD_ON_JETSON ON)
    endif ()
endif ()

if (BUILD_ON_JETSON)
    set(WITH_GPU ON)
    set(ENABLE_TRT ON)
    add_compile_options(-O2 -march=armv8-a -mtune=cortex-a78 -ffast-math -fopenmp)
endif ()


file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" MD_VERSION)
string(STRIP ${MD_VERSION} MD_VERSION)

# 解析主/次/补丁版本（用于 SOVERSION 与 ABI 策略）
string(REPLACE "." ";" MD_VERSION_PARTS ${MD_VERSION})
list(GET MD_VERSION_PARTS 0 MD_VERSION_MAJOR)
list(LENGTH MD_VERSION_PARTS MD_VERSION_PARTS_LEN)
if (MD_VERSION_PARTS_LEN GREATER 1)
    list(GET MD_VERSION_PARTS 1 MD_VERSION_MINOR)
else ()
    set(MD_VERSION_MINOR 0)
endif ()
if (MD_VERSION_PARTS_LEN GREATER 2)
    list(GET MD_VERSION_PARTS 2 MD_VERSION_PATCH)
else ()
    set(MD_VERSION_PATCH 0)
endif ()

add_definitions(-DMD_VERSION="${MD_VERSION}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TRT_DIR "" CACHE PATH "TRT_DIR")
message(STATUS "TRT_DIR=${TRT_DIR}")

if (POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif ()


# /utf-8 对 C 与 CXX 都生效（samplerate 等 C 源文件否则触发 C4819 文件代码页刷屏）；
# 不作用到 CUDA（nvcc 裸 /utf-8 会触发 single-input 报错，CUDA 经下方 -Xcompiler /utf-8 传入）
if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
    add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4251>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4244>")
    add_definitions(-DMD_CXX_EXPORT)
    if (NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
        set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
    endif ()
else ()
    # GCC/Clang：第三方库已通过 md_silence_external 静默，自有代码的告警级别在
    # ModelDeploySDK 目标上单独开启（见 md_warn_own），避免告警选项向下泄漏到第三方。
    add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>")
    # SIMD 优化（仅 x86_64；aarch64 用 NEON/SVE，编译器自动启用，-mavx2 不适用）
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        add_compile_options(
                "$<$<COMPILE_LANGUAGE:CXX>:-mavx2>"
                "$<$<COMPILE_LANGUAGE:CXX>:-mfma>")
    endif ()
endif ()

# 对第三方库自身目标静默全部警告（如 zxing/kaldi/samplerate/Catch2 自带的 -Wall、-Wextra、/W4 等）。
# 目标添加了目录级警告选项，target 级 -w(/w) 追加在命令行末尾可将其覆盖。
function(md_silence_external target)
    if (TARGET ${target})
        if (MSVC)
            target_compile_options(${target} PRIVATE /w)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -w)
        endif ()
    endif ()
endfunction()

# 对自有 SDK 代码开启合理告警级别（不含 -Werror）。target 级生效，避免泄漏到第三方。
# /W3 仅作用 C/CXX；作用到 CUDA 会让 nvcc 报 single-input 致命错（CUDA 无独立告警级别需求）。
function(md_warn_own target)
    if (MSVC)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W3>")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif ()
endfunction()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/csrc)
# 第三方头一律按系统头处理：自有代码编译时不再泄漏第三方头文件内的警告
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party)


file(GLOB_RECURSE ALL_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/*.cpp)
file(GLOB_RECURSE ORT_BACKEND_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/ort/*.cpp)
file(GLOB_RECURSE MNN_BACKEND_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/mnn/*.cpp)
file(GLOB_RECURSE TRT_BACKEND_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/trt/*.cpp)
file(GLOB_RECURSE SOPHGO_BACKEND_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/sophgo/*.cpp
        ${CMAKE_SOURCE_DIR}/csrc/vision/processors/sophgo/*.cpp)
file(GLOB_RECURSE VISION_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/vision/*.cpp)
file(GLOB_RECURSE AUDIO_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/audio/*.cpp)
file(GLOB_RECURSE VIDEO_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/video/*.cpp)
file(GLOB_RECURSE PYBIND_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/pybind/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/csrc/vision/processors/cuda/*.cu
        ${CMAKE_SOURCE_DIR}/csrc/vision/face/face_det/*.cu
        ${CMAKE_SOURCE_DIR}/csrc/vision/processors/cuda/*.cpp)

list(REMOVE_ITEM ALL_SOURCE ${ORT_BACKEND_SOURCE} ${MNN_BACKEND_SOURCE} ${TRT_BACKEND_SOURCE} ${SOPHGO_BACKEND_SOURCE} ${VISION_SOURCE} ${AUDIO_SOURCE} ${PYBIND_SOURCE} ${CUDA_SOURCE} ${VIDEO_SOURCE})

if (WITH_GPU)
    add_definitions(-DWITH_GPU)
    enable_language(CUDA)
    # CUDA 架构：默认 native（自动探测本机 GPU），可用 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif ()
    # for cudaMalloc()
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_MAJOR_VERSION ${CUDAToolkit_VERSION_MAJOR})
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    if (MSVC)
        # -diag-suppress=1394 抑制c++标准库导出警告
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8 -diag-suppress=1394")
    endif ()
    list(APPEND ALL_SOURCE ${CUDA_SOURCE})
    list(APPEND DEPENDS CUDA::cudart)
endif ()


if (ENABLE_ORT)
    add_definitions(-DENABLE_ORT)
    include("${CMAKE_SOURCE_DIR}/cmake/onnxruntime.cmake")
    list(APPEND ALL_SOURCE ${ORT_BACKEND_SOURCE})
    list(APPEND DEPENDS onnxruntime::onnxruntime)
endif ()

if (ENABLE_MNN)
    add_definitions(-DENABLE_MNN)
    include("${CMAKE_SOURCE_DIR}/cmake/mnn.cmake")
    list(APPEND ALL_SOURCE ${MNN_BACKEND_SOURCE})
    list(APPEND DEPENDS ${MNN_LIBS})
endif ()

if (ENABLE_TRT)
    if (NOT WITH_GPU)
        message(FATAL_ERROR "When ENABLE_TRT=ON, must set WITH_GPU=ON")
    endif ()
    if (APPLE)
        message(FATAL_ERROR "Apple does not support TRT backend")
    else ()
        add_definitions(-DENABLE_TRT)
        include("${CMAKE_SOURCE_DIR}/cmake/trt.cmake")
        list(APPEND ALL_SOURCE ${TRT_BACKEND_SOURCE})
        list(APPEND DEPENDS trt::nvinfer)
        list(APPEND DEPENDS trt::nvonnxparser)
    endif ()
endif ()

if (ENABLE_SOPHGO)
    if (WIN32)
        message(FATAL_ERROR "SOPHGO backend is Linux-only (Sophon-Sail has no Windows build)")
    endif ()
    include("${CMAKE_SOURCE_DIR}/cmake/sophgo.cmake")
    if (SOPHGO_FOUND)
        list(APPEND ALL_SOURCE ${SOPHGO_BACKEND_SOURCE})
        list(APPEND DEPENDS ${SOPHGO_LIBS})
    else ()
        message(STATUS "Sophon-Sail not found, SOPHGO backend disabled")
    endif ()
endif ()


if (BUILD_VISION)
    add_definitions(-DBUILD_VISION)
    include("${CMAKE_SOURCE_DIR}/cmake/opencv.cmake")
    # visualize 使用 OpenCV 5 的 cv::FontFace / getTextSize 新签名；OpenCV 4 下排除（避免编译失败）
    if (OpenCV_VERSION VERSION_LESS "5.0")
        list(FILTER VISION_SOURCE EXCLUDE REGEX ".*/visualize/.*\\.cpp$")
        message(STATUS "OpenCV < 5: visualize module excluded")
    endif ()
    # sophgo 预处理后端仅随 ENABLE_SOPHGO 编译（见下方 ENABLE_SOPHGO 块）
    list(REMOVE_ITEM VISION_SOURCE ${SOPHGO_BACKEND_SOURCE})
    # CUDA 预处理后端仅随 WITH_GPU 编译（见上方 WITH_GPU 块）
    list(REMOVE_ITEM VISION_SOURCE ${CUDA_SOURCE})
    list(APPEND ALL_SOURCE ${VISION_SOURCE})
    list(APPEND DEPENDS ${OpenCV_LIBS})
endif ()

if (BUILD_AUDIO)
    add_definitions(-DBUILD_AUDIO)
    set(BUILD_TESTING OFF CACHE BOOL "Disable testing for samplerate" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/samplerate)
    md_silence_external(samplerate)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Disable Python binding for kaldi-native-fbank" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Disable testing for kaldi-native-fbank" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "With shared library for kaldi-native-fbank" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank EXCLUDE_FROM_ALL)
    md_silence_external(kaldi-native-fbank-core)
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank)
    list(APPEND ALL_SOURCE ${AUDIO_SOURCE})
    # 音频静态依赖 PRIVATE 链接（打进共享库，导出时无需第三方 target）
    list(APPEND PRIVATE_DEPENDS samplerate kaldi-native-fbank-core)
    if (ENABLE_WETEXT)
        # 可选 WeTextProcessing ITN 后端（覆盖最全）；需第三方 OpenFst + we-text-processing。
        # 通过 WETEXT_INCLUDE_DIR / WETEXT_FST_DIR 传入 include 与模型目录。
        find_path(WETEXT_INCLUDE_DIR "we-text-processing/csrc/cn_itn_processor.h")
        if (WETEXT_INCLUDE_DIR)
            include_directories(${WETEXT_INCLUDE_DIR})
            add_definitions(-DMODELDEPLOY_HAS_WETEXT)
            message(STATUS "WeTextProcessing ITN backend: ENABLED")
        else ()
            message(WARNING "ENABLE_WETEXT=ON 但未找到 WeTextProcessing 头文件，ITN 退化到轻量实现。"
                            "请设置 -DWETEXT_INCLUDE_DIR=<we-text-processing 根目录>")
        endif ()
    endif ()
endif ()

# ── NLP ──────────────────────────────────────
# cppjieba 同时被音频（Kokoro TTS 分词）与 NLP 使用，需在任一启用时构建；
# 但 -DBUILD_NLP 宏仅在 BUILD_NLP 打开时定义，保证 NLP 扩展（pybind/capi/tests）可独立隔离。
if (BUILD_AUDIO OR BUILD_NLP)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include)
    list(APPEND PRIVATE_DEPENDS cppjieba)
endif ()
if (BUILD_NLP)
    add_definitions(-DBUILD_NLP)
endif ()

# ── VIDEO ──────────────────────────────────
if (BUILD_VIDEO)
    if (NOT BUILD_VISION)
        message(FATAL_ERROR "BUILD_VIDEO requires BUILD_VISION=ON (VideoDecoder returns vision::ImageData)")
    endif ()
    if (ENABLE_GSTREAMER)
        include("${CMAKE_SOURCE_DIR}/cmake/gstreamer.cmake")
        if (NOT GSTREAMER_INCLUDE_DIR OR NOT GSTREAMER_LIBS)
            set(ENABLE_GSTREAMER OFF)
        endif ()
    endif ()
    if (NOT ENABLE_FFMPEG AND NOT ENABLE_GSTREAMER)
        message(FATAL_ERROR "BUILD_VIDEO requires ENABLE_FFMPEG=ON or ENABLE_GSTREAMER=ON")
    endif ()
    if (ENABLE_FFMPEG)
        include("${CMAKE_SOURCE_DIR}/cmake/ffmpeg.cmake")
        if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)   # 沿用现有优雅禁用
            set(ENABLE_FFMPEG OFF)
        endif ()
    endif ()
endif ()
if (BUILD_VIDEO)
    add_definitions(-DBUILD_VIDEO)
    list(APPEND ALL_SOURCE ${VIDEO_SOURCE})
    if (ENABLE_FFMPEG)
        add_definitions(-DENABLE_FFMPEG)
        include_directories(${FFMPEG_INCLUDE_DIR})
        list(APPEND DEPENDS ${FFMPEG_LIBS})
    endif ()
    if (ENABLE_GSTREAMER)
        add_definitions(-DENABLE_GSTREAMER)
        if (GSTREAMER_HAS_CUDA)
            add_definitions(-DHAVE_GSTCUDA)
            # 设备直编/直通的 D2D 拷贝：用 CUDA 驱动 API（cuda.h→nvcuda.dll 的 cuda.lib），
            # 最小化链接，避免把整库卷进 cudart 运行时。
            add_definitions(-DMODELDEPLOY_CUDA_DRV)
            get_filename_component(_md_cuda_root "${GSTREAMER_CUDA_INCLUDE}" DIRECTORY)
            find_library(MD_CUDA_DRIVER_LIB cuda PATHS "${_md_cuda_root}/lib/x64" NO_DEFAULT_PATH)
            if (MD_CUDA_DRIVER_LIB)
                list(APPEND DEPENDS ${MD_CUDA_DRIVER_LIB})
            endif()
        endif ()
        if (GSTREAMER_HAS_NVBUF)
            add_definitions(-DHAVE_NVBUF)
        endif ()
        include_directories(${GSTREAMER_INCLUDE_DIR})
        list(APPEND DEPENDS ${GSTREAMER_LIBS})
    endif ()
    # VAAPI 硬解/硬编（Linux + libva，仅编译门控）：默认 OFF，Windows 不定义该宏，相关代码不参与编译。
    # VAAPI 的设备上下文由 FFmpeg 的 libavutil hwcontext_vaapi / GStreamer vaapi 插件承载，
    # libva 由 FFmpeg/GStreamer 自身链接，故此处只需控制编译宏，不额外 find libva。
    # Sophgo 硬解需 sophon-mw（bm_video_decode），当前 cmake/sophgo.cmake 仅提供推理 SDK（bmrt/bmcv/bmlib），
    # 故 Sophgo 真实帧路径在本模块内未接线（仅命中/校验/fail-closed，见 backend 实现注释）。
    if (ENABLE_VAAPI)
        add_definitions(-DENABLE_VAAPI)
    endif ()
endif ()

if (BUILD_BARCODE)
    add_definitions(-DBUILD_BARCODE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ZXING_WRITERS OFF CACHE STRING "" FORCE)
    set(ZXING_READERS ON CACHE STRING "" FORCE)
    # 仅需读取库；禁用 examples（会 FetchContent 拉取 stb，无网络时失败）与 C-API wrapper
    set(ZXING_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ZXING_C_API OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build ZXing as static library" FORCE)
    # ZXing v3.1.1 核心需要 C++20；仅提升 ZXing 子工程的标准，随后恢复 SDK 的 C++17
    set(_md_prev_cxx_std "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 20)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/zxing-cpp EXCLUDE_FROM_ALL)
    md_silence_external(ZXing)
    set(CMAKE_CXX_STANDARD "${_md_prev_cxx_std}")
    unset(_md_prev_cxx_std)
    include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/third_party/zxing-cpp/core/src)
    list(APPEND PRIVATE_DEPENDS ZXing)
endif ()

if (BUILD_CAPI)
    include("${CMAKE_SOURCE_DIR}/capi/CMakeLists.txt")
    if (MSVC)
        add_definitions(-DMD_CAPI)
    endif ()
    list(APPEND ALL_SOURCE ${CAPI_SOURCE})
endif ()

# 模型加密（可选，需要 OpenSSL）
if (BUILD_ENCRYPTION)
    find_package(OpenSSL QUIET)
    if (OpenSSL_FOUND)
        add_definitions(-DENABLE_ENCRYPTION)
        include_directories(${OPENSSL_INCLUDE_DIR})
        list(APPEND DEPENDS OpenSSL::SSL OpenSSL::Crypto)
        message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
    else ()
        message(WARNING "BUILD_ENCRYPTION=ON but OpenSSL not found. "
                "Model encryption will be disabled.\n"
                "  Windows: Download from https://slproweb.com/products/Win32OpenSSL.html\n"
                "           Install the Win64 OpenSSL Light version (e.g. Win64 OpenSSL v3.x.x)\n"
                "           Then set: cmake -DOPENSSL_ROOT_DIR=\"C:/Program Files/OpenSSL-Win64\"\n"
                "  Linux:   sudo apt install libssl-dev  (Debian/Ubuntu)\n"
                "           sudo dnf install openssl-devel  (Fedora/RHEL)\n"
                "           sudo pacman -S openssl  (Arch)\n"
                "  macOS:   brew install openssl")
    endif ()
endif ()

add_library(${LIBRARY_NAME} SHARED ${ALL_SOURCE})
set_target_properties(${LIBRARY_NAME} PROPERTIES
        VERSION ${MD_VERSION}
        SOVERSION ${MD_VERSION_MAJOR})
md_warn_own(${LIBRARY_NAME})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPENDS} PRIVATE ${PRIVATE_DEPENDS})
# Sophgo 平台 jpg 变暗修复：whole-archive 逻辑定义在 cmake/opencv.cmake（MD_OCV_JPEG_TURBO_WHOLE_ARCHIVE）
if (NOT "${MD_OCV_JPEG_TURBO_WHOLE_ARCHIVE}" STREQUAL "")
    target_link_options(${LIBRARY_NAME} PRIVATE ${MD_OCV_JPEG_TURBO_WHOLE_ARCHIVE})
endif ()
if (MSVC AND ENABLE_ORT)
    # delay-load onnxruntime：首次调用时才加载，此时 win_dll_path.cpp 的静态初始化
    # 已用 SetDllDirectory 指向 SDK 同目录，避免被 System32 的旧版 onnxruntime(1.17) 抢占
    target_link_options(${LIBRARY_NAME} PRIVATE
            "/DELAYLOAD:onnxruntime.dll"
            "/DELAYLOAD:onnxruntime_providers_cuda.dll")
    target_link_libraries(${LIBRARY_NAME} PRIVATE delayimp)
endif ()
if (MSVC)
    target_compile_options(${LIBRARY_NAME} PRIVATE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:/wd4251>")
endif ()

if (UNIX AND NOT APPLE)
    # Linux
    set_target_properties(${LIBRARY_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
            BUILD_RPATH "$ORIGIN"
    )
elseif (APPLE)
    # macOS
    set_target_properties(${LIBRARY_NAME} PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_RPATH "@loader_path"
    )
endif ()

if (BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    # pybind 音频绑定依赖 BUILD_AUDIO（cppjieba 等），关闭时排除
    if (NOT BUILD_AUDIO)
        list(REMOVE_ITEM PYBIND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/pybind/audio/kokoro_pybind.cpp
                ${CMAKE_SOURCE_DIR}/csrc/pybind/audio/speaker_verify_pybind.cpp
                ${CMAKE_SOURCE_DIR}/csrc/pybind/audio/audio8_pybind.cpp
                ${CMAKE_SOURCE_DIR}/csrc/pybind/audio/qwen3_tts_pybind.cpp)
    endif ()
    # pybind NLP 绑定依赖 BUILD_NLP；关闭时排除，否则 SDK 侧 nlp/tools/*.cpp
    # 在 #if defined(BUILD_NLP) 下为空、符号缺失，pyd 链接会报 LNK2001（如 Tokenizer）
    if (NOT BUILD_NLP)
        file(GLOB _MD_PYBIND_NLP_SOURCE ${CMAKE_SOURCE_DIR}/csrc/pybind/nlp/*.cpp)
        list(REMOVE_ITEM PYBIND_SOURCE ${_MD_PYBIND_NLP_SOURCE})
    endif ()
    add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/pybind11")
    pybind11_add_module(modeldeploy ${PYBIND_SOURCE})
    target_link_libraries(modeldeploy PUBLIC ${LIBRARY_NAME})
    set_target_properties(modeldeploy PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
    # 安装依赖的第三方动态库
    file(GLOB EXTRA_DLLS
            "${CMAKE_BINARY_DIR}/bin/*.dll"
            "${CMAKE_BINARY_DIR}/bin/*.so"
            "${CMAKE_BINARY_DIR}/bin/*.so.*"
            "${CMAKE_BINARY_DIR}/bin/*.dylib"
    )
    configure_file(${CMAKE_SOURCE_DIR}/python/__init__.py.in ${CMAKE_SOURCE_DIR}/python/modeldeploy/__init__.py)
    install(TARGETS modeldeploy DESTINATION modeldeploy)
    install(TARGETS ${LIBRARY_NAME}
            RUNTIME DESTINATION modeldeploy
            LIBRARY DESTINATION modeldeploy)
    install(FILES ${EXTRA_DLLS} DESTINATION modeldeploy)  # 使用 install 而非 copy
    install(FILES ${CMAKE_SOURCE_DIR}/python/modeldeploy/__init__.py DESTINATION modeldeploy)
    # 构建期用 pybind11-stubgen 生成 .pyi 存根，写回 staging 的 modeldeploy/，随 wheel 发布。
    # stubgen 失败只告警不阻断构建（失败时 _stubgen_rv 非 0，降级为 warning）。
    if (DEFINED Python_EXECUTABLE AND EXISTS "${Python_EXECUTABLE}")
        install(CODE "
            set(_stub_root \"\${CMAKE_INSTALL_PREFIX}\")
            execute_process(COMMAND \"${Python_EXECUTABLE}\" -m pybind11_stubgen modeldeploy --output-dir \"\${_stub_root}\"
                WORKING_DIRECTORY \"\${_stub_root}\"
                RESULT_VARIABLE _stubgen_rv)
            if(NOT _stubgen_rv EQUAL 0)
                message(WARNING \"pybind11-stubgen failed (rc=\${_stubgen_rv}); wheel will be built without .pyi stubs.\")
            endif()
        ")
    else()
        message(WARNING "Python_EXECUTABLE undefined; skipping pyi stub generation.")
    endif()
endif ()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (BUILD_BENCHMARK)
    enable_testing()
    add_subdirectory(benchmark)
endif ()

add_subdirectory(application)


include(GNUInstallDirs)

if (NOT BUILD_PYTHON)
    # 安装头文件（递归所有 .h）
    install(DIRECTORY csrc/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/modeldeploy
            FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

    # 安装库文件（例如 libModelDeploySDK.so / .a / .dll）
    install(TARGETS ModelDeploySDK
            EXPORT ModelDeploySDKTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

    file(GLOB THIRD_LIBS
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*.dll"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*.so"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*.so.*"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/*.dylib"
    )
    install(FILES ${THIRD_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})

    # CMake package config：支持 find_package(ModelDeploySDK)
    install(EXPORT ModelDeploySDKTargets
            FILE ModelDeploySDKTargets.cmake
            NAMESPACE ModelDeploySDK::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelDeploySDK)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
            "${CMAKE_SOURCE_DIR}/cmake/ModelDeploySDKConfig.cmake.in"
            "${CMAKE_BINARY_DIR}/ModelDeploySDKConfig.cmake"
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelDeploySDK)
    write_basic_package_version_file(
            "${CMAKE_BINARY_DIR}/ModelDeploySDKConfigVersion.cmake"
            VERSION ${MD_VERSION}
            COMPATIBILITY SameMajorVersion)
    install(FILES
            "${CMAKE_BINARY_DIR}/ModelDeploySDKConfig.cmake"
            "${CMAKE_BINARY_DIR}/ModelDeploySDKConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ModelDeploySDK)
endif ()

