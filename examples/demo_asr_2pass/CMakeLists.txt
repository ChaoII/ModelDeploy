set(TARGET_NAME "demo_asr_2pass")
add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} PUBLIC ${LIB_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")