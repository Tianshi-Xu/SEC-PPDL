cmake_minimum_required(VERSION 3.10)

project(test)

# add_executable(test_Conv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
# target_link_libraries(test_Conv PUBLIC LinearLayer)

# add_executable(test_tensor ${CMAKE_CURRENT_LIST_DIR}/src/test_tensor.cpp)
# target_link_libraries(test_tensor PUBLIC Datatype)

add_executable(test_relu ${CMAKE_CURRENT_LIST_DIR}/src/TestReLU.cpp)
target_link_libraries(test_relu PUBLIC NonlinearLayer)