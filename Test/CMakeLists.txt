cmake_minimum_required(VERSION 3.10)

project(test)

add_executable(test_Conv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
target_link_libraries(test_Conv PUBLIC LinearLayer HE Datatype)