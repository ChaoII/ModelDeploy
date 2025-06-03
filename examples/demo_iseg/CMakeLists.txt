
add_executable(demo_instance_seg_cxx demo_instance_seg_cxx.cpp)
target_link_libraries(demo_instance_seg_cxx PUBLIC ${LIBRARY_NAME})


add_executable(demo_instance_seg_capi demo_instance_seg_capi.cpp)
target_link_libraries(demo_instance_seg_capi PUBLIC ${LIBRARY_NAME})