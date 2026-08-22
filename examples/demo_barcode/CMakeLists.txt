# 纯 CV 演示，无需后端变体
add_executable(demo_barcode demo_barcode.cpp)
target_link_libraries(demo_barcode PUBLIC ${LIBRARY_NAME})
