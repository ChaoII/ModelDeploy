cmake_minimum_required(VERSION 3.20)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 17)


set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE "")
set(DEPENDS "")

message("==================${CMAKE_BUILD_TYPE}===================")
option(ENABLE_ORT "enable ort backend" ON)
option(ENABLE_MNN "enable mnn backend" ON)
option(ENABLE_TRT "enable mnn backend" OFF)
option(BUILD_AUDIO "build asr tts vad sr" ON)
option(BUILD_VISION "build vision module" ON)
option(BUILD_CAPI "build with capi" ON)
option(BUILD_PYTHON "build with python" ON)
option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build tests" OFF)
option(WITH_STATIC_CRT "build for mt" OFF)
option(WITH_GPU "Enable GPU" ON)
option(BUILD_ON_JETSON "Whether to build fastdeploy on Nvidia Jetson" OFF)

if (NOT BUILD_ON_JETSON)
    if (EXISTS "/etc/nv_tegra_release")
        message(STATUS "detect Jetson")
        set(BUILD_ON_JETSON ON)
    endif ()
endif ()

if (BUILD_ON_JETSON)
    set(WITH_GPU ON)
    set(ENABLE_TRT ON)
    find_package(TBB REQUIRED)
    if (TBB_FOUND)
        list(APPEND DEPENDS TBB::tbb)
    else ()
        message(WARNING "Cant not find tbb library, maybe link error!")
    endif ()
    add_compile_options(-O2 -march=armv8-a -mtune=cortex-a78 -ffast-math -fopenmp)
endif ()


file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" MD_VERSION)
string(STRIP ${MD_VERSION} MD_VERSION)

add_definitions(-DMD_VERSION="${MD_VERSION}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(TRT_DIR "" CACHE PATH "TRT_DIR")
message(STATUS ${TRT_DIR})

if (POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif ()


if (MSVC)
    if (WITH_STATIC_CRT)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    endif ()
    add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4251>"
            "$<$<COMPILE_LANGUAGE:CXX>:/wd4244>"
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_definitions(-DMD_CXX_EXPORT)
    if (NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
        set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
    endif ()
endif ()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/csrc)
include_directories(${CMAKE_SOURCE_DIR}/third_party)


file(GLOB_RECURSE ALL_SOURCE ${CMAKE_SOURCE_DIR}/csrc/*.cpp)
file(GLOB_RECURSE ORT_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/ort/*.cpp)
file(GLOB_RECURSE MNN_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/mnn/*.cpp)
file(GLOB_RECURSE TRT_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/trt/*.cpp)
file(GLOB_RECURSE VISION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/*.cpp)
file(GLOB_RECURSE AUDIO_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/*.cpp)
file(GLOB_RECURSE PYBIND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/pybind/*.cpp)
file(GLOB_RECURSE CUDA_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/common/processors/*.cu)

list(REMOVE_ITEM ALL_SOURCE ${ORT_BACKEND_SOURCE} ${MNN_BACKEND_SOURCE} ${TRT_BACKEND_SOURCE} ${VISION_SOURCE} ${AUDIO_SOURCE} ${PYBIND_SOURCE})

if (WITH_GPU)
    add_definitions(-DWITH_GPU)
    enable_language(CUDA)
    # https://developer.nvidia.com/cuda-gpus
    set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 40 系列示例
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


if (BUILD_VISION)
    add_definitions(-DBUILD_VISION)
    include("${CMAKE_SOURCE_DIR}/cmake/opencv.cmake")
    list(APPEND ALL_SOURCE ${VISION_SOURCE})
    list(APPEND DEPENDS ${OpenCV_LIBS})
endif ()

if (BUILD_AUDIO)
    add_definitions(-DBUILD_AUDIO)
    set(BUILD_TESTING OFF CACHE BOOL "Disable testing for samplerate" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/samplerate)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Disable Python binding for kaldi-native-fbank" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Disable testing for kaldi-native-fbank" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "With shared library for kaldi-native-fbank" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank EXCLUDE_FROM_ALL)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    list(APPEND ALL_SOURCE ${AUDIO_SOURCE})
    list(APPEND DEPENDS samplerate kaldi-native-fbank-core cppjieba)
endif ()

if (BUILD_CAPI)
    include("${CMAKE_SOURCE_DIR}/capi/CMakeLists.txt")
    if (MSVC)
        add_definitions(-DMD_CAPI)
    endif ()
    list(APPEND ALL_SOURCE ${CAPI_SOURCE})
endif ()

add_library(${LIBRARY_NAME} SHARED ${ALL_SOURCE})
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${MD_VERSION})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPENDS})
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
endif ()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


include(GNUInstallDirs)

if (NOT BUILD_PYTHON)
    # 安装头文件（递归所有 .h）
    install(DIRECTORY csrc/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/modeldeploy
            FILES_MATCHING PATTERN "*.h")

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
endif ()

