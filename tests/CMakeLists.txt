
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://gitee.com/ziiwee/Catch2
        GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)
add_executable(TEST1 test_detection.cpp test_ocr.cpp test_face.cpp)
target_link_libraries(TEST1 PRIVATE ${LIB_NAME} Catch2::Catch2WithMain)
add_custom_command(
        TARGET TEST1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/${LIB_NAME}.dll ${CMAKE_BINARY_DIR}/tests
        COMMENT "Copying DLL files to ${CMAKE_BINARY_DIR}/tests"
)
add_test(
        NAME TEST1
        COMMAND $<TARGET_FILE:TEST1> --success
)