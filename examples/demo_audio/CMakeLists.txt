
#add_executable(demo_sense_voice_cxx demo_sense_voice_cxx.cpp)
#target_link_libraries(demo_sense_voice_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_sense_voice_cxx1 demo_sense_voice_cxx1.cpp)
target_link_libraries(demo_sense_voice_cxx1 PUBLIC ${LIBRARY_NAME})

#add_executable(demo_sense_voice_capi demo_sense_voice_capi.cpp)
#target_link_libraries(demo_sense_voice_capi PUBLIC ${LIBRARY_NAME})
#
#add_executable(demo_kokoro_tts_cxx demo_kokoro_tts_cxx.cpp)
#target_link_libraries(demo_kokoro_tts_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_kokoro_tts_cxx1 demo_kokoro_tts_cxx1.cpp)
target_link_libraries(demo_kokoro_tts_cxx1 PUBLIC ${LIBRARY_NAME})


#add_executable(demo_kokoro_tts_capi demo_kokoro_tts_capi.cpp)
#target_link_libraries(demo_kokoro_tts_capi PUBLIC ${LIBRARY_NAME})
