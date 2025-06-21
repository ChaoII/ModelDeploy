message("================start--MD_CAPI================")

if (NOT BUILD_CAPI)
    return()
endif ()

file(GLOB_RECURSE CAPI_SOURCE ${CMAKE_SOURCE_DIR}/capi/*.cpp)
file(GLOB_RECURSE VISION_CAPI_SOURCE ${CMAKE_SOURCE_DIR}/capi/vision/*.cpp)
file(GLOB_RECURSE AUDIO_CAPI_SOURCE ${CMAKE_SOURCE_DIR}/capi/audio/*.cpp)

list(REMOVE_ITEM CAPI_SOURCE ${VISION_CAPI_SOURCE} ${AUDIO_CAPI_SOURCE})

if (BUILD_VISION)
    list(APPEND CAPI_SOURCE ${VISION_CAPI_SOURCE})
endif ()

if (BUILD_AUDIO)
    list(APPEND CAPI_SOURCE ${AUDIO_CAPI_SOURCE})
endif ()

