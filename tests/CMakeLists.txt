# 启用测试
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/Catch2" "${CMAKE_BINARY_DIR}/third_party/Catch2")
# These tests can use the Catch2-provided main
add_executable(test_modeldeploy test_md_image.cpp utils.cpp)
target_link_libraries(test_modeldeploy PRIVATE ${LIBRARY_NAME} Catch2::Catch2WithMain)
add_test(NAME test_modeldeploy COMMAND test_modeldeploy)
set_tests_properties(test_modeldeploy PROPERTIES ENVIRONMENT "TEST_DATA_DIR=${CMAKE_SOURCE_DIR}")