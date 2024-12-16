
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://gitee.com/ziiwee/Catch2
        GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(Catch2)
add_executable(TEST1 test_get_text_position.cpp)
target_link_libraries(TEST1 PRIVATE carplate_sdk Catch2::Catch2WithMain)
add_test(
        NAME TEST1
        COMMAND $<TARGET_FILE:TEST1> --success
)