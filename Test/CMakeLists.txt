cmake_minimum_required(VERSION 3.10)

project(test)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
macro (add_test_operation _name)
	add_executable(${_name} "./src/test_${_name}.cpp")
	# include_directories(../src)
    target_link_libraries(${_name} Operator Threads::Threads OpenMP::OpenMP_CXX) 
	# target_compile_definitions(${_name} PUBLIC USE_VOLE=1)
endmacro()

macro (add_test_relu _name)
	add_executable(${_name} "./src/test_${_name}.cpp")
	# include_directories(../src)
    target_link_libraries(${_name} Operator Non-linear Threads::Threads OpenMP::OpenMP_CXX) 
	# target_compile_definitions(${_name} PUBLIC USE_VOLE=1)
endmacro()

add_test_operation(millionaire)
add_test_relu(ring-relu)

# add_executable(test_HE ${CMAKE_CURRENT_LIST_DIR}/src/test_HE.cpp)
# target_link_libraries(test_HE PUBLIC HE Datatype)
