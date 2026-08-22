add_executable(demo_hand demo_hand.cpp)
target_link_libraries(demo_hand PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
