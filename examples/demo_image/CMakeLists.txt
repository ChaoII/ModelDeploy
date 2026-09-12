# C API demos（依赖 BUILD_CAPI）
if (BUILD_CAPI)
    add_executable(demo_image_from_base64 demo_image_from_base64.cpp)
    target_link_libraries(demo_image_from_base64 PUBLIC ${LIBRARY_NAME})

    add_executable(demo_image_from_bgr24 demo_image_from_bgr24.cpp)
    target_link_libraries(demo_image_from_bgr24 PUBLIC ${LIBRARY_NAME})
endif ()

# C++ vision demo（依赖 BUILD_VISION；本目录仅在 BUILD_VISION 时被加入）
add_executable(demo_image_rotate demo_image_rotate.cpp)
target_link_libraries(demo_image_rotate PUBLIC ${LIBRARY_NAME})
