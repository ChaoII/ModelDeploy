
add_executable(demo_instance_seg_cxx demo_instance_seg_cxx.cpp)
target_link_libraries(demo_instance_seg_cxx PUBLIC ${LIBRARY_NAME})

# Sophgo TPU 后端 demo
if(ENABLE_SOPHGO)
add_executable(demo_iseg_sophgo demo_iseg_sophgo.cpp)
target_link_libraries(demo_iseg_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_instance_seg_capi demo_instance_seg_capi.cpp)
target_link_libraries(demo_instance_seg_capi PUBLIC ${LIBRARY_NAME})