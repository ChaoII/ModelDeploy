cmake_minimum_required(VERSION 3.16)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 17)

set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE "")
set(DEPENDS "")

option(BUILD_AUDIO "build asr tts vad sr" OFF)
option(BUILD_VISION "build vision module" ON)
option(BUILD_FACE "build face" ON)
option(BUILD_CAPI "build with capi" ON)
option(ENABLE_SAVE_LOG "save log to log file" OFF)


if (MSVC)
    add_compile_options(
            $<$<CONFIG:>:/MT> #---------|
            $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
            $<$<CONFIG:Release>:/MT> #--|
            $<$<CONFIG:RelWithDebInfo>:/MT>
            $<$<CONFIG:MinSizeRel>:/MT>
    )
    add_compile_options(/wd4819 /wd4244)
endif ()
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party)

file(GLOB_RECURSE RUNTIME_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/*.cpp)
file(GLOB_RECURSE CORE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/core/*.cpp)
file(GLOB_RECURSE FUNCTION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/function/*.cpp)
file(GLOB_RECURSE UTILS_SOURCE ${CMAKE_SOURCE_DIR}/csrc/utils/*.cpp)
list(APPEND SOURCE ${RUNTIME_SOURCE} ${VISION_SOURCE} ${CORE_SOURCE} ${FUNCTION_SOURCE} ${UTILS_SOURCE})
list(APPEND SOURCE csrc/base_model.cpp)

if (BUILD_VISION)
    # opencv
    set(OpenCV_DIR "D:/opencv-5.x/build/install_mt/x64/vc17/staticlib" CACHE PATH "Path to OpenCV build directory")
    find_package(OpenCV CONFIG REQUIRED)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "build BUILD_VISION depends on opencv,please specifier OpenCV_DIR")
    endif ()
    include_directories(${OpenCV_INCLUDE_DIRS})
    file(GLOB_RECURSE VISION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/*/*.cpp ${CMAKE_SOURCE_DIR}/csrc/vision/*.cpp)
    if (BUILD_FACE)
        set(SEETA_DIR "E:/develop/SeetaFace" CACHE PATH "seeta face directory")
        add_definitions(-DBUILD_FACE)
        include(${CMAKE_SOURCE_DIR}/cmake/face.cmake)
    else ()
        file(GLOB FACE_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/face/*.cpp)
        list(REMOVE_ITEM VISION_SOURCE ${FACE_SOURCE})
    endif ()

    list(APPEND SOURCE ${VISION_SOURCE})
    list(APPEND DEPENDS ${OpenCV_LIBS})
endif ()

#spdlog
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/spdlog")
list(APPEND DEPENDS spdlog::spdlog)
if (ENABLE_SAVE_LOG)
    add_definitions(-DSAVE_LOG)
endif ()

if (BUILD_CAPI)
    include(${CMAKE_SOURCE_DIR}/capi/CMakeLists.txt)
    if (MSVC)
        add_definitions(-DMD_CAPI)
    endif ()
endif ()

# onnxruntime
set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/_deps/onnxruntime-src")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)


add_library(${LIBRARY_NAME} SHARED ${SOURCE})
if (BUILD_AUDIO)
    include("${CMAKE_SOURCE_DIR}/cmake/audio.cmake")
    list(APPEND DEPENDS sherpa-onnx-core)
else ()
    list(APPEND DEPENDS onnxruntime)
endif ()
target_link_libraries(${LIBRARY_NAME} ${DEPENDS})
add_subdirectory(examples)

