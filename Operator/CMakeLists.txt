cmake_minimum_required(VERSION 3.10)
project(Operator_test)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Operator STATIC ${srcs})
# add_library(Operator INTERFACE)
target_include_directories(Operator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(Operator PUBLIC
    OT
)
