cmake_minimum_required(VERSION 3.13)
project(sec_ppdl)

# set(CMAKE_PREFIX_PATH "/lustre/gpu4/home/yeyu/SEC-PPDL/extern/SEAL")
# find_package(SEAL REQUIRED)
# target_link_libraries(sec_ppdl PRIVATE SEAL::seal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/extern/SEAL/build/cmake" ${CMAKE_PREFIX_PATH})
find_package(SEAL REQUIRED)
# include_directories(${CMAKE_SOURCE_DIR}/extern/SEAL/build/native/src)
# include_directories(${CMAKE_SOURCE_DIR}/extern/SEAL/native/src)
# link_directories(${CMAKE_SOURCE_DIR}/extern/SEAL/build/lib)

add_subdirectory(src)
# add_subdirectory(tests)
