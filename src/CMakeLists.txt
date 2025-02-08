cmake_minimum_required(VERSION 3.10)
project(test_conv)
# set(SEAL_DIR "/lustre/gpu4/home/yeyu/SEC-PPDL/extern/SEAL/build/cmake/")
# set(SEAL_DIR "$../extern/SEAL/build/cmake" ${CMAKE_PREFIX_PATH})
find_package(SEAL 4.1.2 EXACT QUIET PATHS "/home/xts/code/SEC-PPDL/extern/SEAL/build/" NO_DEFAULT_PATH)
message(STATUS "${SEAL_FOUND}")
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 4.1.2 was not found")
endif()


add_library(LINEAR ${CMAKE_SOURCE_DIR}/layer/linear/Conv.cpp)
target_link_libraries(LINEAR SEAL::seal)

target_include_directories(LINEAR PRIVATE ${CMAKE_SOURCE_DIR}/datatype)
# target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/layer/linear)
# target_include_directories(test_conv PRIVATE ${CMAKE_SOURCE_DIR}/layer)
target_include_directories(LINEAR PRIVATE ${CMAKE_SOURCE_DIR}/HE)

add_executable(test_conv ${CMAKE_SOURCE_DIR}/layer/linear/TestConv.cpp)
target_link_libraries(test_conv LINEAR)
