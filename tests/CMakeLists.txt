# Catch2 仅测试使用，不放在顶层 CMakeLists.txt
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")

# === 统一测试可执行文件 ===
# 所有测试用例编译为单个二进制，用 Catch2 标签 [core] [image_data]
# [capi] [model] 等区分。运行时按需筛选：
#   test_modeldeploy          # 全部
#   test_modeldeploy [core]   # 仅核心
#   test_modeldeploy ~[gpu]   # 排除 GPU

set(TEST_SOURCES
    test_core.cpp
    test_bugfix.cpp
    test_image_data.cpp
    test_capi.cpp
    test_vision_models.cpp
    test_obb_nms.cpp
    test_processor_accuracy.cpp
    test_insightface.cpp
    test_pipelines.cpp
    test_pipeline.cpp
    test_benchmark_insightface.cpp
    test_encryption.cpp
    test_tracking.cpp
    test_barcode.cpp
    test_hand.cpp
    test_cv_tools.cpp
    test_cv_solutions.cpp
    test_audio_tools.cpp
    test_audio_solutions.cpp
    test_paraformer_streaming.cpp
    test_landmark.cpp
    test_action.cpp
    test_reid.cpp
    test_speaker_verify.cpp
    test_formula_recognition.cpp
    test_doc_to_markdown.cpp
    baseline_compare.cpp
    baseline_utils.cpp
    utils.cpp
)

# NLP 测试仅当 BUILD_NLP 打开时参与编译（NLP 扩展可独立隔离）。
if(BUILD_NLP)
    list(APPEND TEST_SOURCES
        test_nlp_tools.cpp
        test_nlp_solutions.cpp
    )
endif()

# 视频核心类型测试：仅当 BUILD_VIDEO 时编译（make_image_from_planes_view 定义在 video 库中）
if(BUILD_VIDEO)
    list(APPEND TEST_SOURCES test_video_core.cpp)
endif()

# GPU-only 测试源：仅当 CUDA 启用时编译（否则 CPU 下 cuda_runtime.h 缺失导致编译失败）
if(WITH_GPU)
    list(APPEND TEST_SOURCES
        test_nv12_batch_preproc.cpp
        test_benchmark_yolo26n.cpp
    )
endif()

add_executable(test_modeldeploy ${TEST_SOURCES})
target_link_libraries(test_modeldeploy PRIVATE
    ${LIBRARY_NAME}
    Catch2::Catch2WithMain
    ${OpenCV_LIBS})
target_include_directories(test_modeldeploy PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)   # nlohmann json

# 视频解码测试（仅 BUILD_VIDEO；测试源由 Task 6 提供）
if (BUILD_VIDEO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_video_decoder.cpp")
    target_sources(test_modeldeploy PRIVATE test_video_decoder.cpp)
    target_include_directories(test_modeldeploy PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_libraries(test_modeldeploy PRIVATE ${FFMPEG_LIBS})
endif ()

# === 回归基线收集工具 ===
add_executable(baseline_collect baseline_collect.cpp baseline_utils.cpp)
target_link_libraries(baseline_collect PRIVATE
    ${LIBRARY_NAME}
    ${OpenCV_LIBS})
target_include_directories(baseline_collect PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)

add_test(NAME test_modeldeploy COMMAND test_modeldeploy)
set_tests_properties(test_modeldeploy PROPERTIES
    ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
