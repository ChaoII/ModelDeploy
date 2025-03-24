add_subdirectory(demo_image)
add_subdirectory(demo_ocr)
add_subdirectory(demo_classification)
add_subdirectory(demo_detection)
add_subdirectory(demo_faceid)

if (BUILD_FACE)
    add_subdirectory(demo_face)
endif ()

if (BUILD_AUDIO)
    add_subdirectory(demo_audio)
endif ()