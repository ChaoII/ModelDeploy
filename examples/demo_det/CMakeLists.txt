

add_executable(demo_detection_cxx demo_detection_cxx.cpp)
target_link_libraries(demo_detection_cxx PUBLIC ${LIBRARY_NAME})

add_executable(demo_detection_batch demo_detection_batch.cpp)
target_link_libraries(demo_detection_batch PUBLIC ${LIBRARY_NAME})


add_executable(demo_detection_capi demo_detection_capi.cpp)
target_link_libraries(demo_detection_capi PUBLIC ${LIBRARY_NAME})


add_executable(demo_detection_multi_thread demo_detection_multi_thread.cpp)
target_link_libraries(demo_detection_multi_thread PUBLIC ${LIBRARY_NAME})

add_executable(demo_detection_multi_thread_trt demo_detection_multi_thread_trt.cpp)
target_link_libraries(demo_detection_multi_thread_trt PUBLIC ${LIBRARY_NAME})