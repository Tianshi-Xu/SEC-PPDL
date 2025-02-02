cmake_minimum_required(VERSION 3.10)
set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
project(NetIO LANGUAGES CXX)
find_package(SEAL CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_executable(netio ./main.cpp ./NetIO.cpp ./HE.h)
target_link_libraries(netio SEAL::seal)