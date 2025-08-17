cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
project(HE_test LANGUAGES CXX)
set(FATHER_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# find_package(SEAL CONFIG REQUIRED)
find_package(SEAL 4.1.2 EXACT QUIET PATHS "${FATHER_PROJECT_SOURCE_DIR}/Extern/SEAL/build" NO_DEFAULT_PATH)

option(USE_HE_GPU "Use GPU backend for HE" OFF)
message(STATUS "USE_HE_GPU: ${USE_HE_GPU}")
# Phantom: GPU backend for HE
if(USE_HE_GPU)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe")
    find_package(Phantom REQUIRED PATHS "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe/build_phantom")
    message(STATUS "Phantom_FOUND:${Phantom_FOUND}")
    # if(NOT Phantom_FOUND)
    #     message(STATUS "Phantom was not found: install locally")
    #     execute_process(COMMAND ${CMAKE_COMMAND} -S . -B build -DCMAKE_INSTALL_PREFIX=${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe
    #             WORKING_DIRECTORY "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe")
    #     execute_process(COMMAND ${CMAKE_COMMAND} --build build --target install --parallel
    #             WORKING_DIRECTORY "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe")
    #     find_package(Phantom REQUIRED PATHS "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe")
    # endif()
endif()

# Set source files
if(NOT USE_HE_GPU)
    set(srcs 
        src/NetIO.cpp
        src/UnifiedHE.cpp)
else()
    set(srcs 
        src/NetIO.cpp
        src/PhantomWrapper.cpp
        src/UnifiedEvaluator.cpp
        src/UnifiedHE.cpp)
endif()

# Add static library for HE
add_library(HE STATIC ${srcs})

# Add include files for HE
target_include_directories(HE PUBLIC include)

# Link libraries for HE
if(NOT USE_HE_GPU)
    target_link_libraries(HE PUBLIC
        SEAL::seal
        Datatype
        Utils
    )
else()
    target_compile_definitions(HE PUBLIC USE_HE_GPU)
    target_link_libraries(HE PUBLIC
        SEAL::seal
        phantom::Phantom
        CUDA::cudart # <cuda_runtime_api.h>
        Datatype
        Utils
    )
endif()
