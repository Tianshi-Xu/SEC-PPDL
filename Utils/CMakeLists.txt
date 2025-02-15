cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Utils STATIC ${srcs})
target_include_directories(Utils PUBLIC include)

message(STATUS "Utils_FOUND: ${Utils_FOUND}")