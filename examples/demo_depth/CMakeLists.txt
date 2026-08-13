# 深度估计 demo（bmodel -> Sophgo TPU；onnx -> ORT CPU）
add_executable(demo_depth demo_depth_sophgo.cpp)
target_link_libraries(demo_depth PUBLIC ${LIBRARY_NAME})
