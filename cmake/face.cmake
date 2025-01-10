
set(FACE_LINK_DIR "")
if (WIN32)
    set(FACE_LINK_DIR "${SEETA_DIR}/lib/windows_x64")
elseif (APPLE)
    message(FATAL_ERROR "Unsupported MAC OS")
elseif (UNIX)
    #if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    if (CURRENT_OSX_ARCH MATCHES "arm64")
        set(FACE_LINK_DIR "${SEETA_DIR}/lib/linux_aarch64")
    else ()
        set(FACE_LINK_DIR "${SEETA_DIR}/lib/linux_x64")
    endif ()
endif ()


include_directories(${SEETA_DIR}/include)
link_directories(${FACE_LINK_DIR})
list(APPEND DEPENDS
        SeetaFaceLandmarker600
        SeetaFaceRecognizer610
        SeetaFaceDetector600
        SeetaFaceAntiSpoofingX600
        SeetaQualityAssessor300
        SeetaPoseEstimation600
        SeetaGenderPredictor600
        SeetaEyeStateDetector200
        SeetaAgePredictor600)

