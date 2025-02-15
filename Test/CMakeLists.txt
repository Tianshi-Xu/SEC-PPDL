cmake_minimum_required(VERSION 3.10)

project(test)

add_executable(test_HE ${CMAKE_CURRENT_LIST_DIR}/src/test_HE.cpp)
target_link_libraries(test_HE PUBLIC HE Datatype)