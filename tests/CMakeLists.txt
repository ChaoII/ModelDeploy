add_definitions(-DCATCH_CONFIG_ENABLE_BENCHMARKING)

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
    test_image_data.cpp
    test_capi_image.cpp
    test_md_image.cpp
    test_vision_models.cpp
    utils.cpp
)

add_executable(test_modeldeploy ${TEST_SOURCES})
target_link_libraries(test_modeldeploy PRIVATE
    ${LIBRARY_NAME}
    Catch2::Catch2WithMain
    ${OpenCV_LIBS})
target_include_directories(test_modeldeploy PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME test_modeldeploy COMMAND test_modeldeploy)
set_tests_properties(test_modeldeploy PROPERTIES
    ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")
