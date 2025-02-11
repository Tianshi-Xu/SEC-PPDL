cmake_minimum_required(VERSION 3.10)

project(test)

add_executable(TestConv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
target_link_libraries(TestConv PUBLIC HE Datatype LinearLayer)