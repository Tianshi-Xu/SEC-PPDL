cmake_minimum_required(VERSION 3.10)
# project(Datatype_test LANGUAGES CXX)

find_package(SEAL 4.1.2 EXACT QUIET PATHS "${CMAKE_CURRENT_LIST_DIR}/../Extern/SEAL/build" NO_DEFAULT_PATH)
message(STATUS "SEAL_FOUND: ${SEAL_FOUND}")

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Datatype STATIC ${srcs})
target_include_directories(Datatype PUBLIC include)

target_link_libraries(Datatype
    SEAL::seal
)
message(STATUS "Datatype_FOUND: ${Datatype_FOUND}")
# add_executable(Datatype_test ${CMAKE_SOURCE_DIR}/src/test.cpp)
# target_link_libraries(Datatype_test Datatype)