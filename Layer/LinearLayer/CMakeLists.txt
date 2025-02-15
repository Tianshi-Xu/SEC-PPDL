cmake_minimum_required(VERSION 3.10)
project(LinearLayer_test LANGUAGES CXX)


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(LinearLayer STATIC ${srcs})
target_include_directories(LinearLayer PUBLIC include)

target_link_libraries(LinearLayer PUBLIC
    Operator
)

message(STATUS "LinearLayer_FOUND: ${LinearLayer_FOUND}")