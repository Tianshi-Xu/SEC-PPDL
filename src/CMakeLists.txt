cmake_minimum_required(VERSION 3.10)
set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
project(NetIO LANGUAGES CXX)
find_package(SEAL CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



add_executable(test ./main.cpp ./HE/NetIO.cpp)
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/datatype)
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/layer/linear)
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/layer)
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/HE)
target_link_libraries(test SEAL::seal)