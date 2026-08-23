add_executable(demo_solutions demo_solutions.cpp)
target_link_libraries(demo_solutions PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
