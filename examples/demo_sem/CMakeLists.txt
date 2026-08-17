# ����ָ� demo��bmodel -> Sophgo TPU��onnx -> ORT/GPU��
if(ENABLE_SOPHGO)
add_executable(demo_sem_sophgo demo_sem_sophgo.cpp)
target_link_libraries(demo_sem_sophgo PUBLIC ${LIBRARY_NAME})
endif()

add_executable(demo_sem_cxx demo_sem_cxx.cpp)
target_link_libraries(demo_sem_cxx PUBLIC ${LIBRARY_NAME})
