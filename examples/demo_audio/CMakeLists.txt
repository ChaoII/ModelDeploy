
add_executable(demo_sense_voice_cxx demo_sense_voice_cxx.cpp)
target_link_libraries(demo_sense_voice_cxx PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_sense_voice_cxx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(demo_sense_voice_capi demo_sense_voice_capi.cpp)
target_link_libraries(demo_sense_voice_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_sense_voice_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


add_executable(demo_kokoro_tts_cxx demo_kokoro_tts_cxx.cpp)
target_link_libraries(demo_kokoro_tts_cxx PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_kokoro_tts_cxx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(demo_kokoro_tts_capi demo_kokoro_tts_capi.cpp)
target_link_libraries(demo_kokoro_tts_capi PUBLIC ${LIBRARY_NAME})
set_target_properties(demo_kokoro_tts_capi PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")