cmake_minimum_required(VERSION 3.20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Phantom LANGUAGES CXX CUDA VERSION 2.1 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

include(GNUInstallDirs)
include(FetchContent)

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

set(PHANTOM_MIN_RMM_VERSION "25.08")
option(PHANTOM_AUTO_INSTALL_RMM "Automatically install RMM when missing or too old" ON)
set(PHANTOM_RMM_SOURCE_DIR "$ENV{RMM_SOURCE_DIR}" CACHE PATH "Local RMM source root (contains cpp/CMakeLists.txt)")
set(PHANTOM_RMM_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/rmm-build" CACHE PATH "Build directory for auto-installed RMM")
set(PHANTOM_RMM_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_deps/rmm-install" CACHE PATH "Install prefix for auto-installed RMM")

function(phantom_ensure_rmm min_version)
    find_package(rmm ${min_version} CONFIG QUIET)
    if (rmm_FOUND)
        return()
    endif ()

    if (NOT PHANTOM_AUTO_INSTALL_RMM)
        message(FATAL_ERROR "RMM >= ${min_version} is required. Set PHANTOM_AUTO_INSTALL_RMM=ON to auto-install.")
    endif ()

    set(_rmm_candidates
            "${PHANTOM_RMM_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../rmm"
            "$ENV{HOME}/rmm"
            "/data/home/chaoyang/rmm")
    set(_rmm_src "")
    foreach (_candidate IN LISTS _rmm_candidates)
        if (_candidate AND EXISTS "${_candidate}/cpp/CMakeLists.txt")
            set(_rmm_src "${_candidate}")
            break()
        endif ()
    endforeach ()

    if (NOT _rmm_src)
        message(STATUS "RMM source not found locally. Fetching rmm branch-25.08...")
        FetchContent_Declare(
                rmm_src
                GIT_REPOSITORY https://github.com/rapidsai/rmm.git
                GIT_TAG branch-25.08
                GIT_SHALLOW ON)
        FetchContent_Populate(rmm_src)
        set(_rmm_src "${rmm_src_SOURCE_DIR}")
    endif ()

    message(STATUS "Auto-installing RMM from: ${_rmm_src}")
    message(STATUS "RMM build dir: ${PHANTOM_RMM_BUILD_DIR}")
    message(STATUS "RMM install prefix: ${PHANTOM_RMM_INSTALL_PREFIX}")

    set(_rmm_config_cmd
            ${CMAKE_COMMAND}
            -S "${_rmm_src}/cpp"
            -B "${PHANTOM_RMM_BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${PHANTOM_RMM_INSTALL_PREFIX}
            -DBUILD_TESTS=OFF
            -DBUILD_BENCHMARKS=OFF
            -DRMM_NVTX=OFF)
    if (DEFINED CMAKE_CUDA_ARCHITECTURES)
        list(APPEND _rmm_config_cmd -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
    endif ()

    execute_process(
            COMMAND ${_rmm_config_cmd}
            RESULT_VARIABLE _rmm_config_status)
    if (NOT _rmm_config_status EQUAL 0)
        message(FATAL_ERROR "Failed to configure RMM auto-install (status=${_rmm_config_status}).")
    endif ()

    execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${PHANTOM_RMM_BUILD_DIR}" --target install --parallel
            RESULT_VARIABLE _rmm_build_status)
    if (NOT _rmm_build_status EQUAL 0)
        message(FATAL_ERROR "Failed to build/install RMM (status=${_rmm_build_status}).")
    endif ()

    list(PREPEND CMAKE_PREFIX_PATH "${PHANTOM_RMM_INSTALL_PREFIX}")
    find_package(rmm ${min_version} CONFIG REQUIRED PATHS "${PHANTOM_RMM_INSTALL_PREFIX}" NO_DEFAULT_PATH)
endfunction()

phantom_ensure_rmm(${PHANTOM_MIN_RMM_VERSION})
message(STATUS "Use RMM memory pool: ON")
message(STATUS "RMM version requirement: >= ${PHANTOM_MIN_RMM_VERSION} (fixed)")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
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
install(TARGETS Phantom
        EXPORT PhantomConfig
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)
install(EXPORT PhantomConfig NAMESPACE phantom:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phantom)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)
