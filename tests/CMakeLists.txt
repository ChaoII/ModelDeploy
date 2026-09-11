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
    test_mnn_device.cpp
    test_ptr_validate.cpp
    test_bugfix.cpp
    test_image_data.cpp
    test_vision_models.cpp
    test_obb_nms.cpp
    test_processor_accuracy.cpp
    test_insightface.cpp
    test_pipelines.cpp
    test_pipeline.cpp
    test_benchmark_insightface.cpp
    test_encryption.cpp
    test_encryption_e2e.cpp
    test_tracking.cpp
    test_barcode.cpp
    test_hand.cpp
    test_cv_tools.cpp
    test_cv_solutions.cpp
    test_fall_detector.cpp
    test_fall_detector.cpp
    test_fastsam.cpp
    test_landmark.cpp
    test_action.cpp
    test_reid.cpp
    test_formula_recognition.cpp
    test_doc_to_markdown.cpp
    test_result_json.cpp
    baseline_compare.cpp
    baseline_utils.cpp
    utils.cpp
    test_ncnn_backend.cpp
    test_ncnn_vision.cpp
    test_ncnn_phase_a.cpp
    test_ncnn_phase_b.cpp
    test_ncnn_lpr.cpp
    test_ncnn_ocr.cpp
    test_ncnn_seetaface.cpp
    test_async_model.cpp
)

# Serving 测试仅在 BUILD_SERVING 时编译（否则 serving 源不编入 SDK，链接失败）。
if(BUILD_SERVING)
    list(APPEND TEST_SOURCES test_serving.cpp)
endif()

# CAPI 测试：仅当 BUILD_CAPI 打开时参与编译（否则 capi/md_capi.h 缺失 + 库中无 CAPI 符号）。
if(BUILD_CAPI)
    list(APPEND TEST_SOURCES test_capi.cpp)
endif()

# 音频测试：仅当 BUILD_AUDIO 打开时参与编译（依赖 cppjieba、audio/* 源；关闭时库中无对应实现导致链接失败）。
if(BUILD_AUDIO)
    list(APPEND TEST_SOURCES
        test_audio_tools.cpp
        test_audio_solutions.cpp
        test_paraformer_streaming.cpp
        test_speaker_verify.cpp
        test_audio_tts.cpp
    )
endif()

# NLP 测试仅当 BUILD_NLP 打开时参与编译（NLP 扩展可独立隔离）。
if(BUILD_NLP)
    list(APPEND TEST_SOURCES
        test_nlp_tools.cpp
        test_nlp_solutions.cpp
    )
endif()

# 视频核心类型测试：仅当 BUILD_VIDEO 时编译（make_image_from_planes_view 定义在 video 库中）
if(BUILD_VIDEO)
    list(APPEND TEST_SOURCES
        test_video_core.cpp
        test_video_factory.cpp
        test_video_ffmpeg_decode.cpp
        test_video_ffmpeg_encode.cpp
        test_video_pipeline.cpp
        test_video_gpu_decode.cpp
        test_video_gpu_encode.cpp)
    if(ENABLE_GSTREAMER)
        list(APPEND TEST_SOURCES test_video_gst_decode.cpp test_video_gst_encode.cpp)
    endif()
    # VAAPI/Sophgo 硬件路径：配置/选择/回退决策本机可测；真实硬件路径由 tag 门控
    list(APPEND TEST_SOURCES test_video_hw_paths.cpp)
endif()

# GPU-only 测试源：仅当 CUDA 启用时编译（否则 CPU 下 cuda_runtime.h 缺失导致编译失败）
if(WITH_GPU)
    list(APPEND TEST_SOURCES
        test_nv12_batch_preproc.cpp
        test_benchmark_yolo26n.cpp
        test_benchmark_clone.cpp
        test_gpu_vis.cpp
    )
endif()

# Sophgo(TPU) 设备帧中间算子测试：仅当 ENABLE_SOPHGO 且找到 libsophon 时编译
# （测试直接调用 bm_dev_request / bmcv_*，需链接 SOPHON 库）。
# test_sophgo_device_ops.cpp 针对新版 libsophon(bm_image_get_plane_memory 等)编写，
# 与设备/构建环境 libsophon 0.4.14 API 不兼容(另含 bmcv_flip_mode 的 ROTATE_180 全局枚举冲突)，
# 在 0.4.14 下无法编译。该文件为内部测试，非交付物，故从 test_modeldeploy 排除。
# 如需在 0.4.14 上验证 TPU 设备中间算子，请参照 sophgo_processor_backend.cpp 集成。
if(FALSE)
    list(APPEND TEST_SOURCES test_sophgo_device_ops.cpp)
endif()

add_executable(test_modeldeploy ${TEST_SOURCES})
target_link_libraries(test_modeldeploy PRIVATE
    ${LIBRARY_NAME}
    Catch2::Catch2WithMain
    ${OpenCV_LIBS}
    ${SOPHGO_LIBS})
target_include_directories(test_modeldeploy PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)   # nlohmann json

# 设备侧 vis 渲染存图工具（真实图像 → NV12 → 各 vis 绘制 → 回读 PNG）
# CPU 端 NV12 图元绘制性能基准（无 GPU 依赖）
add_executable(test_bench_vis_cpu test_bench_vis_cpu.cpp)
target_link_libraries(test_bench_vis_cpu PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
target_include_directories(test_bench_vis_cpu PRIVATE ${CMAKE_SOURCE_DIR})

if(WITH_GPU)
    add_executable(render_vis_gpu render_vis_gpu.cpp)
    target_link_libraries(render_vis_gpu PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
    target_include_directories(render_vis_gpu PRIVATE ${CMAKE_SOURCE_DIR})
    set(_cu_dir "${CUDAToolkit_LIBRARY_DIR}")
    if(EXISTS "${_cu_dir}/cudart.lib")
        target_link_libraries(render_vis_gpu PRIVATE "${_cu_dir}/cudart.lib")
    endif()

    # 整体(全场景,含 D2H 拷贝)可视化绘制基准：设备端绘制 vs 主机绘制
    add_executable(test_bench_vis_holistic test_bench_vis_holistic.cpp)
    # cudart: Windows 直接链 cudart.lib; Linux 链 CUDA::cudart（libcudart.so），
    # 均不可用时降级为不链（设备端绘制例程仍可编译）。
    if(EXISTS "${_cu_dir}/cudart.lib")
        target_link_libraries(test_bench_vis_holistic PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS} "${_cu_dir}/cudart.lib")
    elseif(TARGET CUDA::cudart)
        target_link_libraries(test_bench_vis_holistic PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS} CUDA::cudart)
    else()
        target_link_libraries(test_bench_vis_holistic PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
    endif()
    target_include_directories(test_bench_vis_holistic PRIVATE ${CMAKE_SOURCE_DIR})
endif()

# 视频解码测试（仅 BUILD_VIDEO；测试源由 Task 6 提供）
if (BUILD_VIDEO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_video_decoder.cpp")
    target_sources(test_modeldeploy PRIVATE test_video_decoder.cpp)
    target_include_directories(test_modeldeploy PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_libraries(test_modeldeploy PRIVATE ${FFMPEG_LIBS})
endif ()

# 视频 GPU 直编测试用 FFmpeg 分配设备 NV12 源 + GStreamer CUDA memory 包装，
# 需直接链接 FFmpeg/GStreamer 符号（DLL 不复导出第三方符号）。
if(BUILD_VIDEO AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_video_gpu_encode.cpp")
    target_include_directories(test_modeldeploy PRIVATE ${FFMPEG_INCLUDE_DIR} ${GSTREAMER_INCLUDE_DIR})
    if(ENABLE_FFMPEG)
        target_link_libraries(test_modeldeploy PRIVATE ${FFMPEG_LIBS})
    endif()
    if(ENABLE_GSTREAMER)
        target_link_libraries(test_modeldeploy PRIVATE ${GSTREAMER_LIBS})
    endif()
    # 设备 NV12 源用 cudaMalloc：CUDA 可用（gstcuda 或 WITH_GPU）时定义测试宏并链接 cudart。
    # 否则该测试文件整体 SKIP（不引 cuda_runtime.h 到 CPU 构建）。
    if(GSTREAMER_HAS_CUDA OR WITH_GPU)
        target_compile_definitions(test_modeldeploy PRIVATE MODELDEPLOY_TEST_CUDA)
        set(_cuda_lib_dir "${CUDAToolkit_LIBRARY_DIR}")
        if(EXISTS "${_cuda_lib_dir}/cudart.lib")
            target_link_libraries(test_modeldeploy PRIVATE "${_cuda_lib_dir}/cudart.lib")
        endif()
        unset(_cuda_lib_dir)
    endif()
endif()

# === 回归基线收集工具 ===
add_executable(baseline_collect baseline_collect.cpp baseline_utils.cpp)
target_link_libraries(baseline_collect PRIVATE
    ${LIBRARY_NAME}
    ${OpenCV_LIBS})
target_include_directories(baseline_collect PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/application/third_party)

add_test(NAME test_modeldeploy COMMAND test_modeldeploy ~[opencl] ~[vulkan])
set_tests_properties(test_modeldeploy PROPERTIES
    ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
