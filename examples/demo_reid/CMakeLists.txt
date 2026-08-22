add_executable(demo_reid demo_reid.cpp)
target_link_libraries(demo_reid PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
