cmake_minimum_required(VERSION 3.10)
project(Model_test LANGUAGES CXX)



file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(OT STATIC ${srcs})
target_include_directories(OT PUBLIC include)

target_link_libraries(OT PUBLIC
    Datatype
)

message(STATUS "OT_FOUND: ${OT_FOUND}")