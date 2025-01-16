
add_executable(demo_asr_offline_file demo_asr_offline_file.cpp)
target_link_libraries(demo_asr_offline_file PUBLIC ${LIB_NAME})
set_target_properties(demo_asr_offline_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


add_executable(demo_asr_2pass_buffer demo_asr_2pass_buffer.cpp)
target_link_libraries(demo_asr_2pass_buffer PUBLIC ${LIB_NAME})
set_target_properties(demo_asr_2pass_buffer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_compile_features(demo_asr_2pass_buffer PUBLIC cxx_std_11)