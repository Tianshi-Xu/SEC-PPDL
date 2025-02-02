cmake_minimum_required(VERSION 3.10)
set(SEAL_DIR "/lustre/gpu4/home/yeyu/SEC-PPDL/extern/SEAL/build/cmake/")
project(test_conv)

find_package(SEAL 4.1.1)
add_library(LINEAR ${CMAKE_SOURCE_DIR}/layer/linear/Conv.cpp)

add_executable(test_conv ${CMAKE_SOURCE_DIR}/layer/linear/TestConv.cpp)
target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/datatype)
target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/layer/linear)
target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/layer)
target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/HE)
target_link_libraries(test_conv SEAL::seal LINEAR)
