add_executable(demo_fastsam demo_fastsam.cpp)
target_link_libraries(demo_fastsam PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
