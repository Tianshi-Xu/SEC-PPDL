cmake_minimum_required(VERSION 3.10)

project(SEC_PPDL)
set(CMAKE_CXX_STANDARD 17)            # 使用C++17
set(CMAKE_CXX_STANDARD_REQUIRED True) # 强制使用指定的C++标准
set(CMAKE_CXX_EXTENSIONS OFF)        # 禁用编译器特定的扩展（如GNU扩展）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(Datatype)
add_subdirectory(HE)
add_subdirectory(OT)
add_subdirectory(Utils)
add_subdirectory(Operator)
add_subdirectory(Layer)
add_subdirectory(Test)
