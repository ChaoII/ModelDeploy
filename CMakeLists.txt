cmake_minimum_required(VERSION 3.16)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 20)


set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE "")
set(DEPENDS "")

message("==================${CMAKE_BUILD_TYPE}===================")

option(BUILD_AUDIO "build asr tts vad sr" ON)
option(BUILD_VISION "build vision module" ON)
option(BUILD_CAPI "build with capi" ON)
option(BUILD_PYTHON "build with python" ON)
option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build tests" OFF)
option(WITH_STATIC_CRT "build for mt" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    list(APPEND OpenMP::OpenMP_CXX)
else ()
    message(WARNING "OpenMP not found. Compilation will be without OpenMP support.")
endif ()

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
    add_definitions(-DBUILD_VISION)
    # opencv
    include("${CMAKE_SOURCE_DIR}/cmake/opencv.cmake")
    # for debug
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OpenCV_DIR "E:/develop/opencv/build/x64/vc16/lib")
    endif ()
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
    add_definitions(-DBUILD_AUDIO)
    file(GLOB_RECURSE ASR_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/asr/*.cpp)
    file(GLOB_RECURSE VAD_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/vad/*.cpp)
    file(GLOB_RECURSE TTS_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/tts/*.cpp ${CMAKE_SOURCE_DIR}/csrc/audio/tts/*/*.cpp)
    file(GLOB_RECURSE ASR_PIPELINE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/*.cpp)
    set(BUILD_TESTING OFF CACHE BOOL "Disable testing for samplerate" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/samplerate)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Disable Python binding for kaldi-native-fbank" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Disable testing for kaldi-native-fbank" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "With shared library for kaldi-native-fbank" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/kaldi_native_fbank EXCLUDE_FROM_ALL)
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
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPENDS})


if (BUILD_PYTHON)
    file(GLOB_RECURSE PYBIND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/pybind/*.cpp)
    add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/pybind11")
    pybind11_add_module(modeldeploy ${PYBIND_SOURCE})
    target_link_libraries(modeldeploy PUBLIC ${LIBRARY_NAME})
    install(TARGETS modeldeploy DESTINATION modeldeploy)
    install(TARGETS ${LIBRARY_NAME}
            RUNTIME DESTINATION modeldeploy  # .dll
            LIBRARY DESTINATION modeldeploy)

    # 安装依赖的第三方动态库
    file(GLOB EXTRA_DLLS
            "${CMAKE_BINARY_DIR}/bin/*.dll"
            "${CMAKE_BINARY_DIR}/bin/*.so"
            "${CMAKE_BINARY_DIR}/bin/*.dylib"
    )

    foreach (dll_file ${EXTRA_DLLS})
        add_custom_command(TARGET modeldeploy POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll_file}"
                "${CMAKE_SOURCE_DIR}/python/modeldeploy/"
                COMMENT "Copying ${dll_file} to python/modeldeploy/"
        )
    endforeach ()
endif ()


if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


