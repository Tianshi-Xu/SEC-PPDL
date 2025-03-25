cmake_minimum_required(VERSION 3.10)

project(test)

# add_executable(test_HE ${CMAKE_CURRENT_LIST_DIR}/src/test_HE.cpp)
# target_link_libraries(test_HE PUBLIC HE Utils)

add_executable(test_linear_operator ${CMAKE_CURRENT_LIST_DIR}/src/TestLinearOperator.cpp)
target_link_libraries(test_linear_operator PUBLIC LinearOperator Utils)

# add_executable(test_Conv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
# target_link_libraries(test_Conv PUBLIC Model)

# add_executable(test_tensor ${CMAKE_CURRENT_LIST_DIR}/src/test_tensor.cpp)
# target_link_libraries(test_tensor PUBLIC Datatype)

# add_executable(test_relu ${CMAKE_CURRENT_LIST_DIR}/src/TestReLU2.cpp)
# target_link_libraries(test_relu PUBLIC Model)

# add_executable(test_fixpoint ${CMAKE_CURRENT_LIST_DIR}/src/TestFixPoint.cpp)
# target_link_libraries(test_fixpoint PUBLIC NonlinearOperator)

# add_executable(test_resnet ${CMAKE_CURRENT_LIST_DIR}/src/TestResNet.cpp)
# target_link_libraries(test_resnet PUBLIC Model)

# add_executable(test_pool ${CMAKE_CURRENT_LIST_DIR}/src/test_Pool.cpp)
# target_link_libraries(test_pool PUBLIC NonlinearLayer)