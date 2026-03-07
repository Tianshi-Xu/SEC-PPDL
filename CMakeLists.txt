cmake_minimum_required(VERSION 3.10)

project(SEC_PPDL)
set(CMAKE_CXX_STANDARD 17)            # 使用C++17
set(CMAKE_CXX_STANDARD_REQUIRED True) # 强制使用指定的C++标准
set(CMAKE_CXX_EXTENSIONS OFF)        # 禁用编译器特定的扩展（如GNU扩展）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EMP-TOOL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Extern/emp-tool")
set(EMP-TOOL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/Extern/emp-tool")
set(EMP-OT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Extern/emp-ot")
# message(STATUS "EMP-TOOL Directory: ${EMP-TOOL_INCLUDE_DIR}")
# message(STATUS "emp tool found ${EMP-TOOL_FOUND}")
# message(STATUS "emp ot found ${EMP-OT_FOUND}")
include_directories(${EMP-OT_INCLUDE_DIR} 
                    ${EMP-TOOL_INCLUDE_DIR}) 

# CUDA 13+ may place cuda::std headers under "<cuda-root>/.../include/cccl".
# Add it globally in GPU builds so every target can resolve <cuda/std/...>.
if(USE_HE_GPU)
    set(_cuda_cccl_candidates
        "$ENV{CUDA_HOME}/targets/x86_64-linux/include/cccl"
        "$ENV{CUDA_HOME}/include/cccl"
        "/usr/local/cuda/targets/x86_64-linux/include/cccl"
        "/usr/local/cuda/include/cccl")
    foreach(_cccl_dir IN LISTS _cuda_cccl_candidates)
        if(_cccl_dir AND EXISTS "${_cccl_dir}")
            include_directories(SYSTEM "${_cccl_dir}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${_cccl_dir}")
            message(STATUS "Global CUDA CCCL include: ${_cccl_dir}")
            break()
        endif()
    endforeach()
endif()


add_subdirectory(Datatype)
add_subdirectory(Utils)
add_subdirectory(HE)
add_subdirectory(OT)
add_subdirectory(Operator)
add_subdirectory(Layer)
add_subdirectory(Model)
add_subdirectory(Test)
