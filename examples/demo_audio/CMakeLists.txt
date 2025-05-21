add_executable(demo_sense_voice_cxx demo_sense_voice_cxx.cpp)
target_link_libraries(demo_sense_voice_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_kokoro_cxx demo_kokoro_cxx.cpp)
target_link_libraries(demo_kokoro_cxx PUBLIC ${LIBRARY_NAME})