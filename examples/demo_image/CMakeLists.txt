
add_executable(demo_image_from_base64 demo_image_from_base64.cpp)
target_link_libraries(demo_image_from_base64 PUBLIC ${LIBRARY_NAME})


add_executable(demo_image_from_bgr24 demo_image_from_bgr24.cpp)
target_link_libraries(demo_image_from_bgr24 PUBLIC ${LIBRARY_NAME})