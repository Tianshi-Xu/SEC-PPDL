cmake_minimum_required(VERSION 3.10)

project(SEC_PPDL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add_subdirectory(Datatype)
# add_subdirectory(HE)
# add_subdirectory(Operator)
# add_subdirectory(Layer)


add_subdirectory(Utils)
add_subdirectory(OT)
add_subdirectory(Operator)
add_subdirectory(Non-linear)
add_subdirectory(Test)