cmake_minimum_required(VERSION 3.10)
project(LinearLayer_test LANGUAGES CXX)

find_package(SEAL 4.1.2 EXACT QUIET PATHS "${CMAKE_CURRENT_LIST_DIR}/../Extern/SEAL/build" NO_DEFAULT_PATH)
message(STATUS "SEAL_FOUND: ${SEAL_FOUND}")

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(LinearLayer STATIC ${srcs})
target_include_directories(LinearLayer PUBLIC include)

target_link_libraries(LinearLayer PUBLIC
    SEAL::seal
    Datatype
    Operator
    HE
)

message(STATUS "LinearLayer_FOUND: ${LinearLayer_FOUND}")