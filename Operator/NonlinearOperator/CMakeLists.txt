cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
# find_package(SEAL CONFIG REQUIRED)
project(NonlinearOperator_test)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(NonlinearOperator STATIC ${srcs})
target_include_directories(NonlinearOperator PUBLIC include)

target_link_libraries(NonlinearOperator PUBLIC
    OTProtocol
)
