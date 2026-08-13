# 语义分割 demo（bmodel -> Sophgo TPU；onnx -> ORT CPU）
add_executable(demo_sem demo_sem_sophgo.cpp)
target_link_libraries(demo_sem PUBLIC ${LIBRARY_NAME})
