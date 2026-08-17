# ��ȹ��� demo��bmodel -> Sophgo TPU��onnx -> ORT/GPU��
if(ENABLE_SOPHGO)
add_executable(demo_depth_sophgo demo_depth_sophgo.cpp)
target_link_libraries(demo_depth_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_depth_cxx demo_depth_cxx.cpp)
target_link_libraries(demo_depth_cxx PUBLIC ${LIBRARY_NAME})
