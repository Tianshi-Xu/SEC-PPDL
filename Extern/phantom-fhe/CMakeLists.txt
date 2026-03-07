cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Phantom LANGUAGES CXX CUDA VERSION 2.1 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

# [option] PHANTOM_USE_CUDA_PTX (default: ON)
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif ()

# [option] FP64_MM_ARITH (default: OFF)
option(FP64_MM_ARITH "Use FP64 arithmetic for modular multiplication" OFF)
message(STATUS "Use FP64 arithmetic for modular multiplication: ${FP64_MM_ARITH}")
if (FP64_MM_ARITH)
    add_compile_definitions(FP64_MM_ARITH)
endif ()

# [option] RNS_POLY_BATCH (default: ON)
option(RNS_POLY_BATCH "Use RNS polynomial batch processing" ON)
message(STATUS "Use RNS polynomial batch processing: ${RNS_POLY_BATCH}")
if (RNS_POLY_BATCH)
    add_compile_definitions(RNS_POLY_BATCH)
endif ()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# library
add_subdirectory(src)

# example
# [option] PHANTOM_ENABLE_EXAMPLE (default: ON)
option(PHANTOM_ENABLE_EXAMPLE "Enable examples" ON)
message(STATUS "Enable examples: ${PHANTOM_ENABLE_EXAMPLE}")
if (PHANTOM_ENABLE_EXAMPLE)
    add_subdirectory(examples)
endif ()

# benchmark
# [option] PHANTOM_ENABLE_BENCH (default: OFF)
option(PHANTOM_ENABLE_BENCH "Enable benchmarks" OFF)
message(STATUS "Enable benchmarks: ${PHANTOM_ENABLE_BENCH}")
if (PHANTOM_ENABLE_BENCH)
    add_subdirectory(benchmark)
endif ()

# test
# [option] PHANTOM_ENABLE_TEST (default: OFF)
option(PHANTOM_ENABLE_TEST "Enable tests" OFF)
message(STATUS "Enable tests: ${PHANTOM_ENABLE_TEST}")
if (PHANTOM_ENABLE_TEST)
    add_subdirectory(test)
endif ()

# python binding
# [option] PHANTOM_ENABLE_PYTHON_BINDING (default: OFF)
option(PHANTOM_ENABLE_PYTHON_BINDING "Enable python binding" OFF)
message(STATUS "Enable python binding: ${PHANTOM_ENABLE_PYTHON_BINDING}")
if (PHANTOM_ENABLE_PYTHON_BINDING)
    add_subdirectory(python)
endif ()

# config for installation
# NOTE: Why is it necessary to explicitly set the CMAKE-INSTALL_XX variable here ?
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include")
install(TARGETS Phantom
        EXPORT PhantomConfig
        INCLUDES DESTINATION include/phantom)
install(EXPORT PhantomConfig NAMESPACE phantom:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phantom)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)
