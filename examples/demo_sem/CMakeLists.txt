# 语义分割 demo（bmodel -> Sophgo TPU；onnx -> ORT/GPU）
add_executable(demo_sem demo_sem_sophgo.cpp)
target_link_libraries(demo_sem PUBLIC ${LIBRARY_NAME})

add_executable(demo_sem_cxx demo_sem_cxx.cpp)
target_link_libraries(demo_sem_cxx PUBLIC ${LIBRARY_NAME})
