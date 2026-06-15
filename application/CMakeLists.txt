# 应用：AI 智能安防监控平台
# 通过顶层 CMake 的 BUILD_SURVEILLANCE=ON 激活

if (NOT BUILD_SURVEILLANCE)
    return()
endif ()

message(STATUS "Building AI Surveillance Application")

# ── FFmpeg ──────────────────────────────────
set(FFMPEG_ROOT "E:/develop/ffmpeg" CACHE PATH "FFmpeg installation root")
find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h PATHS ${FFMPEG_ROOT}/include NO_DEFAULT_PATH)
foreach (lib avcodec avformat avutil avdevice swscale)
    find_library(FFMPEG_${lib}_LIBRARY NAMES ${lib} ${lib}.lib
            PATHS ${FFMPEG_ROOT}/lib NO_DEFAULT_PATH)
    if (FFMPEG_${lib}_LIBRARY)
        list(APPEND FFMPEG_LIBS ${FFMPEG_${lib}_LIBRARY})
    endif ()
endforeach ()
if (NOT FFMPEG_INCLUDE_DIR OR NOT FFMPEG_LIBS)
    message(FATAL_ERROR "FFmpeg not found at ${FFMPEG_ROOT}")
endif ()

# cpp-httplib + nlohmann-json (header-only)
set(HTTPLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ── 应用源文件 ──────────────────────────────
set(APP_SOURCES
        main.cpp
        config.cpp
        pipeline_manager.cpp
        pipeline.cpp
        stream_decoder.cpp
        stream_hub.cpp
        inference_engine.cpp
        infer_group.cpp
        draw_engine.cpp
        stream_encoder.cpp
        http_server.cpp
        perf_stats.cpp
        frame_pool.cpp
)

add_executable(surveillance ${APP_SOURCES})
target_include_directories(surveillance PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HTTPLIB_INCLUDE_DIR}
        ${FFMPEG_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(surveillance PRIVATE
        ${LIBRARY_NAME}
        ${OpenCV_LIBS}
        ${FFMPEG_LIBS}
        CUDA::cudart
)

# ── 测试（可选） ──────────────────────────────
option(BUILD_SURVEILLANCE_TESTS "build surveillance tests" OFF)
if (BUILD_SURVEILLANCE_TESTS)
    if (NOT TARGET Catch2::Catch2WithMain)
        add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")
    endif ()

    # 测试用源文件 = 除 main.cpp 外的所有应用源文件 + 测试源文件
    list(FILTER APP_SOURCES EXCLUDE REGEX "main\\.cpp$")
    set(TEST_SOURCES tests/test_config.cpp
            tests/test_perf_stats.cpp
            tests/test_frame_pool.cpp
            tests/test_stream_decoder.cpp
            tests/test_inference_engine.cpp
            tests/test_infer_group.cpp
            tests/test_draw_engine.cpp
            tests/test_stream_encoder.cpp
            tests/test_pipeline.cpp
            tests/test_pipeline_manager.cpp
            tests/test_http_server.cpp)
    add_executable(surveillance_test ${TEST_SOURCES} ${APP_SOURCES})
    target_include_directories(surveillance_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${HTTPLIB_INCLUDE_DIR}
            ${FFMPEG_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(surveillance_test PRIVATE
            ${LIBRARY_NAME} ${OpenCV_LIBS} ${FFMPEG_LIBS}
            CUDA::cudart Catch2::Catch2WithMain
    )
    add_test(NAME surveillance_test COMMAND surveillance_test)
endif ()
