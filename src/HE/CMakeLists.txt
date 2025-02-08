cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
project(NetIO LANGUAGES CXX)
# find_package(SEAL CONFIG REQUIRED)
find_package(SEAL 4.1.2 EXACT QUIET PATHS "/home/xts/code/SEC-PPDL/extern/SEAL/build/" NO_DEFAULT_PATH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_executable(netio ./main.cpp ./NetIO.cpp ./HE.h)
target_link_libraries(netio SEAL::seal)