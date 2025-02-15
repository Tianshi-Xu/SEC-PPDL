cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
# find_package(SEAL CONFIG REQUIRED)
project(Operator_test)
find_package(SEAL 4.1.2 EXACT QUIET PATHS "${CMAKE_CURRENT_LIST_DIR}/../Extern/SEAL/build" NO_DEFAULT_PATH)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Operator STATIC ${srcs})
target_include_directories(Operator PUBLIC include)

target_link_libraries(Operator PUBLIC
    SEAL::seal
    Datatype
    HE
)
