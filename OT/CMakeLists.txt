cmake_minimum_required(VERSION 3.10)
project(Model_test LANGUAGES CXX)


# file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
# add_library(OT STATIC ${srcs})
# target_include_directories(OT PUBLIC include)


# file(GLOB_RECURSE hdrs CONFIGURE_DEPENDS include/*.h)
add_library(OT INTERFACE)
target_include_directories(OT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "OT_CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")


target_link_libraries(OT
    INTERFACE Utils
)

message(STATUS "OT_FOUND: ${OT_FOUND}")