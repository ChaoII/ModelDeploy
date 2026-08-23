add_executable(demo_landmark demo_landmark.cpp)
target_link_libraries(demo_landmark PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
