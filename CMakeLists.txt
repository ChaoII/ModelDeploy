cmake_minimum_required(VERSION 3.16)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 20)

set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE ""
        csrc/audio/tts/utils.h)
set(DEPENDS "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_AUDIO "build asr tts vad sr" ON)
option(BUILD_VISION "build vision module" ON)
option(BUILD_CAPI "build with capi" ON)
option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build tests" ON)
option(WITH_STATIC_CRT "build for mt" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    if (WITH_STATIC_CRT)
        add_compile_options(
                $<$<CONFIG:>:/MT>
                $<$<CONFIG:Debug>:/MTd>
                $<$<CONFIG:Release>:/MT>
                $<$<CONFIG:RelWithDebInfo>:/MT>
                $<$<CONFIG:MinSizeRel>:/MT>
        )
    endif ()
    add_compile_options(/wd4819 /wd4244)
    add_definitions(-DMD_CXX_EXPORT)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    if (NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
        set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
    endif ()
endif ()
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party)

include("${CMAKE_SOURCE_DIR}/cmake/onnxruntime.cmake")
file(GLOB_RECURSE RUNTIME_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/*.cpp)
file(GLOB_RECURSE CORE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/core/*.cpp)
file(GLOB_RECURSE FUNCTION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/function/*.cpp)
file(GLOB_RECURSE UTILS_SOURCE ${CMAKE_SOURCE_DIR}/csrc/utils/*.cpp)
list(APPEND SOURCE ${RUNTIME_SOURCE} ${VISION_SOURCE} ${CORE_SOURCE} ${FUNCTION_SOURCE} ${UTILS_SOURCE})
list(APPEND SOURCE csrc/base_model.cpp)

if (BUILD_VISION)
    # opencv
    include("${CMAKE_SOURCE_DIR}/cmake/opencv.cmake")
    find_package(OpenCV CONFIG REQUIRED)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "build BUILD_VISION depends on opencv,please specifier OpenCV_DIR")
    endif ()
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    file(GLOB_RECURSE VISION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/*/*.cpp ${CMAKE_SOURCE_DIR}/csrc/vision/*.cpp)
    list(APPEND SOURCE ${VISION_SOURCE})
    list(APPEND DEPENDS ${OpenCV_LIBS})
endif ()

if (BUILD_AUDIO)
    file(GLOB_RECURSE ASR_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/asr/*.cpp)
    file(GLOB_RECURSE VAD_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/vad/*.cpp)
    file(GLOB_RECURSE TTS_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/tts/*.cpp ${CMAKE_SOURCE_DIR}/csrc/audio/tts/*/*.cpp)
    file(GLOB_RECURSE ASR_PIPELINE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/*.cpp)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/samplerate)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Disable Python binding for kaldi-native-fbank" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Disable testing for kaldi-native-fbank" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "With shared library for kaldi-native-fbank" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    list(APPEND SOURCE ${ASR_SOURCE} ${VAD_SOURCE} ${TTS_SOURCE} ${ASR_PIPELINE_SOURCE})
    list(APPEND DEPENDS samplerate kaldi-native-fbank-core cppjieba)
endif ()


if (BUILD_CAPI)
    include(${CMAKE_SOURCE_DIR}/capi/CMakeLists.txt)
    if (MSVC)
        add_definitions(-DMD_CAPI)
    endif ()
endif ()

add_library(${LIBRARY_NAME} SHARED ${SOURCE})
target_link_libraries(${LIBRARY_NAME} ${DEPENDS})

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
