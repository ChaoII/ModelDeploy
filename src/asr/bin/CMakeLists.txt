set(TARGET_NAME "asr_wss_server")

if (WIN32)
    add_definitions(-D_WEBSOCKETPP_CPP11_RANDOM_DEVICE_)
    add_definitions(-D_WEBSOCKETPP_CPP11_TYPE_TRAITS_)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif ()

include_directories("${CMAKE_SOURCE_DIR}/src/asr/bin")
include_directories("${CMAKE_SOURCE_DIR}/src/asr/internal")
include_directories("${CMAKE_SOURCE_DIR}/3rd_party/asio/asio/include")
include_directories("${CMAKE_SOURCE_DIR}/3rd_party/websocketpp")
include_directories("${CMAKE_SOURCE_DIR}/3rd_party/json/include")

set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Vcpkg toolchain file")
message("${CMAKE_TOOLCHAIN_FILE}")
find_package(OpenSSL REQUIRED)


file(GLOB ASR_WSS_SERVER_SOURCE "${CMAKE_SOURCE_DIR}/src/asr/bin/*.cpp")
add_executable(${TARGET_NAME} ${ASR_WSS_SERVER_SOURCE})
target_include_directories(${TARGET_NAME} PUBLIC ${OPENSSL_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC ${LIB_NAME} ${OPENSSL_CRYPTO_LIBRARY} ${OPENSSL_SSL_LIBRARY})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")