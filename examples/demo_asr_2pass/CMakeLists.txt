set(TARGET_NAME "demo_asr_2pass")
add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} PUBLIC ${LIB_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


add_executable(demo_asr_2pass_buffer demo_asr_2pass_buffer.cpp)
target_link_libraries(demo_asr_2pass_buffer PUBLIC ${LIB_NAME})
set_target_properties(demo_asr_2pass_buffer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")