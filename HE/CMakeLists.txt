cmake_minimum_required(VERSION 3.10)
# set(SEAL_DIR "/root/IOTest/seal/SEAL/build/cmake/")
project(HE_test LANGUAGES CXX)
include(FetchContent)
set(FATHER_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# find_package(SEAL CONFIG REQUIRED)
find_package(SEAL 4.1.2 EXACT QUIET PATHS "${FATHER_PROJECT_SOURCE_DIR}/Extern/SEAL/build" NO_DEFAULT_PATH)

option(USE_HE_GPU "Use GPU backend for HE" OFF)
set(HE_MIN_RMM_VERSION "25.08")
option(HE_AUTO_INSTALL_RMM "Automatically install RMM when missing or too old" ON)
set(HE_RMM_SOURCE_DIR "$ENV{RMM_SOURCE_DIR}" CACHE PATH "Local RMM source root (contains cpp/CMakeLists.txt)")
set(HE_RMM_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/rmm-build" CACHE PATH "Build directory for auto-installed RMM")
set(HE_RMM_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_deps/rmm-install" CACHE PATH "Install prefix for auto-installed RMM")
option(HE_AUTO_INSTALL_PHANTOM "Automatically install Phantom package when missing or not RMM-enabled" ON)
set(HE_PHANTOM_SOURCE_DIR "${FATHER_PROJECT_SOURCE_DIR}/Extern/phantom-fhe" CACHE PATH "Phantom source root")
set(HE_PHANTOM_BUILD_DIR "${HE_PHANTOM_SOURCE_DIR}/build" CACHE PATH "Build directory for Phantom package install")
set(HE_PHANTOM_INSTALL_PREFIX "${HE_PHANTOM_SOURCE_DIR}/build_phantom" CACHE PATH "Install prefix for Phantom package")
message(STATUS "USE_HE_GPU: ${USE_HE_GPU}")
# Phantom: GPU backend for HE
if(USE_HE_GPU)
    function(he_ensure_rmm min_version)
        find_package(rmm ${min_version} CONFIG QUIET)
        if(rmm_FOUND)
            return()
        endif()

        if(NOT HE_AUTO_INSTALL_RMM)
            message(FATAL_ERROR "RMM >= ${min_version} is required. Set HE_AUTO_INSTALL_RMM=ON to auto-install.")
        endif()

        set(_rmm_candidates
            "${HE_RMM_SOURCE_DIR}"
            "${FATHER_PROJECT_SOURCE_DIR}/../rmm"
            "$ENV{HOME}/rmm"
            "/data/home/chaoyang/rmm")
        set(_rmm_src "")
        foreach(_candidate IN LISTS _rmm_candidates)
            if(_candidate AND EXISTS "${_candidate}/cpp/CMakeLists.txt")
                set(_rmm_src "${_candidate}")
                break()
            endif()
        endforeach()

        if(NOT _rmm_src)
            message(STATUS "RMM source not found locally. Fetching rmm branch-25.08...")
            FetchContent_Declare(
                rmm_src
                GIT_REPOSITORY https://github.com/rapidsai/rmm.git
                GIT_TAG branch-25.08
                GIT_SHALLOW ON
            )
            FetchContent_Populate(rmm_src)
            set(_rmm_src "${rmm_src_SOURCE_DIR}")
        endif()

        message(STATUS "Auto-installing RMM from: ${_rmm_src}")
        message(STATUS "RMM build dir: ${HE_RMM_BUILD_DIR}")
        message(STATUS "RMM install prefix: ${HE_RMM_INSTALL_PREFIX}")

        set(_rmm_config_cmd
            ${CMAKE_COMMAND}
            -S "${_rmm_src}/cpp"
            -B "${HE_RMM_BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${HE_RMM_INSTALL_PREFIX}
            -DBUILD_TESTS=OFF
            -DBUILD_BENCHMARKS=OFF
            -DRMM_NVTX=OFF
        )
        if(DEFINED CMAKE_CUDA_ARCHITECTURES)
            list(APPEND _rmm_config_cmd -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
        endif()

        execute_process(
            COMMAND ${_rmm_config_cmd}
            RESULT_VARIABLE _rmm_config_status
        )
        if(NOT _rmm_config_status EQUAL 0)
            message(FATAL_ERROR "Failed to configure RMM auto-install (status=${_rmm_config_status}).")
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${HE_RMM_BUILD_DIR}" --target install --parallel
            RESULT_VARIABLE _rmm_build_status
        )
        if(NOT _rmm_build_status EQUAL 0)
            message(FATAL_ERROR "Failed to build/install RMM (status=${_rmm_build_status}).")
        endif()

        list(PREPEND CMAKE_PREFIX_PATH "${HE_RMM_INSTALL_PREFIX}")
        find_package(rmm ${min_version} CONFIG REQUIRED PATHS "${HE_RMM_INSTALL_PREFIX}" NO_DEFAULT_PATH)
    endfunction()

    find_package(CUDAToolkit REQUIRED)
    he_ensure_rmm(${HE_MIN_RMM_VERSION})
    message(STATUS "RMM version requirement: >= ${HE_MIN_RMM_VERSION} (fixed)")

    function(he_ensure_phantom)
        set(_need_phantom_build FALSE)
        set(_phantom_config_file "${HE_PHANTOM_INSTALL_PREFIX}/lib/cmake/phantom/PhantomConfig.cmake")

        if(NOT EXISTS "${_phantom_config_file}")
            set(_need_phantom_build TRUE)
        else()
            file(READ "${_phantom_config_file}" _phantom_config_text)
            if(NOT _phantom_config_text MATCHES "rmm::rmm")
                message(STATUS "Installed Phantom package does not export rmm::rmm linkage. Rebuilding Phantom...")
                set(_need_phantom_build TRUE)
            endif()
        endif()

        if(_need_phantom_build)
            if(NOT HE_AUTO_INSTALL_PHANTOM)
                message(FATAL_ERROR "Phantom package is missing or not RMM-enabled. Set HE_AUTO_INSTALL_PHANTOM=ON to auto-install.")
            endif()
            message(STATUS "Auto-installing Phantom from: ${HE_PHANTOM_SOURCE_DIR}")
            message(STATUS "Phantom build dir: ${HE_PHANTOM_BUILD_DIR}")
            message(STATUS "Phantom install prefix: ${HE_PHANTOM_INSTALL_PREFIX}")

            set(_phantom_config_cmd
                ${CMAKE_COMMAND}
                    -S "${HE_PHANTOM_SOURCE_DIR}"
                    -B "${HE_PHANTOM_BUILD_DIR}"
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_PREFIX=${HE_PHANTOM_INSTALL_PREFIX}
                    -DPHANTOM_ENABLE_EXAMPLE=OFF
                    -DPHANTOM_ENABLE_BENCH=OFF
                    -DPHANTOM_ENABLE_TEST=OFF
                    -DPHANTOM_ENABLE_PYTHON_BINDING=OFF
                    -DPHANTOM_AUTO_INSTALL_RMM=ON
                    -DPHANTOM_RMM_SOURCE_DIR=${HE_RMM_SOURCE_DIR}
                    -DPHANTOM_RMM_BUILD_DIR=${HE_RMM_BUILD_DIR}
                    -DPHANTOM_RMM_INSTALL_PREFIX=${HE_RMM_INSTALL_PREFIX}
            )
            if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
                list(APPEND _phantom_config_cmd -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
            endif()

            execute_process(
                COMMAND ${_phantom_config_cmd}
                RESULT_VARIABLE _phantom_config_status
            )
            if(NOT _phantom_config_status EQUAL 0)
                message(FATAL_ERROR "Failed to configure Phantom auto-install (status=${_phantom_config_status}).")
            endif()

            execute_process(
                COMMAND ${CMAKE_COMMAND} --build "${HE_PHANTOM_BUILD_DIR}" --target install --parallel
                RESULT_VARIABLE _phantom_build_status
            )
            if(NOT _phantom_build_status EQUAL 0)
                message(FATAL_ERROR "Failed to build/install Phantom (status=${_phantom_build_status}).")
            endif()
        endif()

        find_package(Phantom REQUIRED PATHS "${HE_PHANTOM_INSTALL_PREFIX}" NO_DEFAULT_PATH)
        get_target_property(_phantom_links phantom::Phantom INTERFACE_LINK_LIBRARIES)
        if(NOT _phantom_links OR NOT "${_phantom_links}" MATCHES "rmm::rmm")
            message(FATAL_ERROR "Installed Phantom package is not RMM-enabled (missing rmm::rmm export).")
        endif()
        message(STATUS "Phantom linkage: ${_phantom_links}")
    endfunction()

    he_ensure_phantom()
    # CUDA 13+ may place cuda::std headers under "<cuda include>/cccl".
    set(CUDA_CCCL_INCLUDE_DIRS "")
    foreach(_cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
        if(EXISTS "${_cuda_inc}/cccl")
            list(APPEND CUDA_CCCL_INCLUDE_DIRS "${_cuda_inc}/cccl")
        endif()
    endforeach()
    if(CUDA_CCCL_INCLUDE_DIRS)
        message(STATUS "Adding CUDA CCCL include dirs: ${CUDA_CCCL_INCLUDE_DIRS}")
    endif()
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
    if(CUDA_CCCL_INCLUDE_DIRS)
        target_include_directories(HE PUBLIC ${CUDA_CCCL_INCLUDE_DIRS})
    endif()
    target_link_libraries(HE PUBLIC
        SEAL::seal
        phantom::Phantom
        CUDA::cudart # <cuda_runtime_api.h>
        Datatype
        Utils
    )
endif()
