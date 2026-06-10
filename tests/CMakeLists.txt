add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")
add_definitions(-DCATCH_CONFIG_ENABLE_BENCHMARKING)

# === 核心库测试 ===
add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
add_test(NAME test_core COMMAND test_core)

# === 图像测试（需要 vision + CAPI） ===
if (BUILD_VISION)
    add_executable(test_image_data test_image_data.cpp)
    target_link_libraries(test_image_data PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain ${OpenCV_LIBS})
    add_test(NAME test_image_data COMMAND test_image_data)
    set_tests_properties(test_image_data PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")

    add_executable(test_capi_image test_capi_image.cpp)
    target_link_libraries(test_capi_image PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
    add_test(NAME test_capi_image COMMAND test_capi_image)
    set_tests_properties(test_capi_image PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")

    add_executable(test_vision_models test_vision_models.cpp)
    target_link_libraries(test_vision_models PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
    add_test(NAME test_vision_models COMMAND test_vision_models)
    set_tests_properties(test_vision_models PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")

    add_executable(test_md_image test_md_image.cpp)
    target_link_libraries(test_md_image PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
    add_test(NAME test_md_image COMMAND test_md_image)
    set_tests_properties(test_md_image PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
endif ()

# === 音频模型测试 ===
if (BUILD_AUDIO)
    add_executable(test_audio_models test_audio_models.cpp)
    target_link_libraries(test_audio_models PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
    add_test(NAME test_audio_models COMMAND test_audio_models)
    set_tests_properties(test_audio_models PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
endif ()
