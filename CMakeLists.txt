cmake_minimum_required(VERSION 3.16)
project(ModelDeploySDK)
set(CMAKE_CXX_STANDARD 20)


set(LIBRARY_NAME "ModelDeploySDK")
set(SOURCE "")
set(DEPENDS "")

message("==================${CMAKE_BUILD_TYPE}===================")
option(ENABLE_ORT "enable ort backend" ON)
option(ENABLE_MNN "enable mnn backend" ON)
option(BUILD_AUDIO "build asr tts vad sr" ON)
option(BUILD_VISION "build vision module" ON)
option(BUILD_CAPI "build with capi" ON)
option(BUILD_PYTHON "build with python" ON)
option(BUILD_EXAMPLES "build examples" ON)
option(BUILD_TESTS "build tests" OFF)
option(WITH_STATIC_CRT "build for mt" OFF)
option(WITH_GPU "Enable GPU" OFF)


if(WITH_GPU)
    add_definitions(-DWITH_GPU)
endif ()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif ()

if (MSVC)
    if (WITH_STATIC_CRT)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    endif ()
    add_compile_options(/wd4819 /wd4244 /utf-8)
    add_definitions(-DMD_CXX_EXPORT)
    if (NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
        set(DIST_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
    endif ()
endif ()
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third_party)


file(GLOB_RECURSE ALL_SOURCE ${CMAKE_SOURCE_DIR}/csrc/*.cpp)
file(GLOB_RECURSE ORT_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/ort/*.cpp)
file(GLOB_RECURSE MNN_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/runtime/backends/mnn/*.cpp)
file(GLOB_RECURSE VISION_SOURCE ${CMAKE_SOURCE_DIR}/csrc/vision/*.cpp)
file(GLOB_RECURSE AUDIO_SOURCE ${CMAKE_SOURCE_DIR}/csrc/audio/*.cpp)
file(GLOB_RECURSE PYBIND_SOURCE ${CMAKE_SOURCE_DIR}/csrc/pybind/*.cpp)

list(REMOVE_ITEM ALL_SOURCE ${ORT_BACKEND_SOURCE} ${MNN_BACKEND_SOURCE} ${VISION_SOURCE} ${AUDIO_SOURCE} ${PYBIND_SOURCE})


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
    list(APPEND DEPENDS MNN)
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
target_link_libraries(${LIBRARY_NAME} PUBLIC ${DEPENDS})

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
            "${CMAKE_BINARY_DIR}/bin/*.dylib"
    )
    install(TARGETS modeldeploy DESTINATION python/modeldeploy)
    install(TARGETS ${LIBRARY_NAME}
            RUNTIME DESTINATION python/modeldeploy
            LIBRARY DESTINATION python/modeldeploy)
    install(FILES ${EXTRA_DLLS} DESTINATION python/modeldeploy)  # 使用 install 而非 copy
endif ()


if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


