add_executable(demo_tools demo_tools.cpp)
target_link_libraries(demo_tools PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
