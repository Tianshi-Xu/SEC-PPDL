cmake_minimum_required(VERSION 3.10)
project(Non-linear_test)

# file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
# add_library(Non-linear STATIC ${srcs})
add_library(Non-linear INTERFACE)
target_include_directories(Non-linear INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(Non-linear INTERFACE
    OT
)
