# 应用：AI 智能安防监控平台
# 通过顶层 CMake 的 BUILD_SURVEILLANCE=ON 激活

# ── 模型演示后端（demo_server） ──────────────────────────────
# 独立于 BUILD_SURVEILLANCE：注册 ServingServer 之后端（det/cls 等真实模型族），
# 由 BUILD_WEB_DEMO（默认 OFF，需显式 -DBUILD_WEB_DEMO=ON 开启）门控。
# web 资产（application/web_demo）POST_BUILD 拷到构建目录的 bin/web_demo。
if (BUILD_WEB_DEMO AND BUILD_SERVING)
    add_executable(demo_server demo_server.cpp)
    target_link_libraries(demo_server PRIVATE ${LIBRARY_NAME})
    target_include_directories(demo_server PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
    add_custom_command(TARGET demo_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/application/web_demo ${CMAKE_BINARY_DIR}/bin/web_demo
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/bin/application/demo_labels
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/application/demo_manifest.json
                ${CMAKE_BINARY_DIR}/bin/application/demo_manifest.json
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/application/demo_labels
                ${CMAKE_BINARY_DIR}/bin/application/demo_labels)
endif ()

if (NOT BUILD_SURVEILLANCE)
    return()
endif ()

message(STATUS "Building AI Surveillance Application")

# ── FFmpeg ──────────────────────────────────
# 统一来源（SDK 设计的核心）：直接复用 SDK 侧 cmake/ffmpeg.cmake 已解析的
# FFMPEG_INCLUDE_DIR / FFMPEG_LIBS，app 不再各自 find 一套 —— 同一进程内只一个 ffmpeg，
# 规避双 libavcodec 符号冲突；平台差异（桌面备选 / Jetson / Sophgo-sophon-ffmpeg）由 SDK 层选。
# 仅当 SDK 未启用视频（BUILD_VIDEO=OFF）时兜底 pkg-config。
if (NOT FFMPEG_LIBS)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(FFMPEG QUIET libavcodec libavformat libavutil libswscale)
        if (FFMPEG_FOUND)
            set(FFMPEG_INCLUDE_DIR "${FFMPEG_INCLUDE_DIRS}")
            set(FFMPEG_LIBS "${FFMPEG_LIBRARIES}")
        endif ()
    endif ()
endif ()
if (NOT FFMPEG_LIBS)
    message(FATAL_ERROR "FFmpeg not found (SDK video module should provide FFMPEG_LIBS)")
endif ()

# cpp-httplib + nlohmann-json (header-only)
set(HTTPLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ── 应用源文件 ──────────────────────────────
set(APP_SOURCES
        main.cpp
        config.cpp
        pipeline_manager.cpp
        pipeline.cpp
        video_codec.cpp
        video_sink.cpp
        inference_engine.cpp
        infer_group.cpp
        draw_engine.cpp
        video_source.cpp
        http_server.cpp
        perf_stats.cpp
)

add_executable(surveillance ${APP_SOURCES})
target_include_directories(surveillance PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HTTPLIB_INCLUDE_DIR}
        ${FFMPEG_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
if (WITH_GPU)
    target_include_directories(surveillance PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif ()
target_link_libraries(surveillance PRIVATE
        ${LIBRARY_NAME}
        ${OpenCV_LIBS}
        ${FFMPEG_LIBS}
)
if (WITH_GPU)
    target_link_libraries(surveillance PRIVATE CUDA::cudart)
endif ()

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
            tests/test_inference_engine.cpp
            tests/test_infer_group.cpp
            tests/test_draw_engine.cpp
            tests/test_pipeline.cpp
            tests/test_pipeline_manager.cpp
            tests/test_http_server.cpp
            tests/test_video_codec.cpp
            tests/test_video_source.cpp
            tests/test_video_sink.cpp)
    add_executable(surveillance_test ${TEST_SOURCES} ${APP_SOURCES})
    target_include_directories(surveillance_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${HTTPLIB_INCLUDE_DIR}
            ${FFMPEG_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )
    if (WITH_GPU)
        target_include_directories(surveillance_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif ()
    target_link_libraries(surveillance_test PRIVATE
            ${LIBRARY_NAME} ${OpenCV_LIBS} ${FFMPEG_LIBS}
            Catch2::Catch2WithMain
    )
    if (WITH_GPU)
        target_link_libraries(surveillance_test PRIVATE CUDA::cudart)
    endif ()
    add_test(NAME surveillance_test COMMAND surveillance_test)
endif ()
