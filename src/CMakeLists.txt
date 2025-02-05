cmake_minimum_required(VERSION 3.10)
set(SEAL_DIR "/lustre/gpu4/home/yeyu/SEC-PPDL/extern/SEAL/build/cmake/")
project(test_conv)

find_package(SEAL 4.1.1 REQUIRED)
add_library(LINEAR ${CMAKE_SOURCE_DIR}/layer/linear/Conv.cpp)

add_executable(test_conv ${CMAKE_SOURCE_DIR}/layer/linear/TestConv.cpp ${CMAKE_SOURCE_DIR}/HE/NetIO.cpp)

include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR})
include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/operator)
include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/layer/linear)
include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/../extern/SEAL/native/src)
target_link_libraries(test_conv SEAL::seal LINEAR)
