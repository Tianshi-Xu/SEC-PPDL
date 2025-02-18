cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
# find_package(SEAL CONFIG REQUIRED)
project(Operator_test)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Operator STATIC ${srcs})
target_include_directories(Operator PUBLIC include)

target_link_libraries(Operator PUBLIC
    Datatype
    HE
    OT
)
