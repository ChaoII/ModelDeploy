# 深度估计 demo（bmodel -> Sophgo TPU；onnx -> ORT/GPU）
add_executable(demo_depth_sophgo demo_depth_sophgo.cpp)
target_link_libraries(demo_depth_sophgo PUBLIC ${LIBRARY_NAME})

add_executable(demo_depth_cxx demo_depth_cxx.cpp)
target_link_libraries(demo_depth_cxx PUBLIC ${LIBRARY_NAME})
