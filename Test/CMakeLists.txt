cmake_minimum_required(VERSION 3.10)

project(test)

# add_executable(test_HE ${CMAKE_CURRENT_LIST_DIR}/src/test_HE.cpp)
# target_link_libraries(test_HE PUBLIC HE Utils)
add_executable(test_HE ${CMAKE_CURRENT_LIST_DIR}/src/test_HE.cpp)
target_link_libraries(test_HE PUBLIC Model)
add_executable(test_pir_baseline ${CMAKE_CURRENT_LIST_DIR}/src/TestPIRBaseline.cpp)
target_link_libraries(test_pir_baseline PUBLIC HE Datatype)


if(USE_HE_GPU)
    add_executable(testHEGPU ${CMAKE_CURRENT_LIST_DIR}/src/TestHEGPU.cpp)
    target_link_libraries(testHEGPU PUBLIC HE Datatype)
    add_executable(testFFN ${CMAKE_CURRENT_LIST_DIR}/src/TestFFN.cpp)
    target_link_libraries(testFFN PUBLIC HE Datatype)
    add_executable(test_pir ${CMAKE_CURRENT_LIST_DIR}/src/TestPIRGPU.cpp)
    target_link_libraries(test_pir PUBLIC PIR)
    add_executable(test_pir_gpu ${CMAKE_CURRENT_LIST_DIR}/src/TestPIRGPU.cpp)
    target_link_libraries(test_pir_gpu PUBLIC PIR)
    add_executable(test_batch_pir_gpu ${CMAKE_CURRENT_LIST_DIR}/src/batch_gpu_pir.cpp)
    target_link_libraries(test_batch_pir_gpu PUBLIC PIR)
    add_executable(test_ntt_microbench_gpu ${CMAKE_CURRENT_LIST_DIR}/src/TestPhantomNTTMicroBench.cpp)
    target_link_libraries(test_ntt_microbench_gpu PUBLIC HE Datatype)

    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        find_package(GTest QUIET)
    endif()
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(test_phantom_batch_gtest
        ${CMAKE_CURRENT_LIST_DIR}/src/TestBatchViewGTest.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/TestBatchOpsGTest.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/TestBsgsMacGTest.cpp)
    target_include_directories(test_phantom_batch_gtest BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/Extern/phantom-fhe/include)
    target_link_libraries(test_phantom_batch_gtest PRIVATE PIR GTest::gtest_main)
endif()


add_executable(test_linear_operator ${CMAKE_CURRENT_LIST_DIR}/src/TestLinearOperator.cpp)
target_link_libraries(test_linear_operator PUBLIC LinearOperator Utils)

add_executable(test_Conv ${CMAKE_CURRENT_LIST_DIR}/src/TestConv.cpp)
target_link_libraries(test_Conv PUBLIC Model)

add_executable(test_matmul ${CMAKE_CURRENT_LIST_DIR}/src/TestMatmul.cpp)
target_link_libraries(test_matmul PUBLIC LinearLayer Model)

add_executable(test_cir_linear ${CMAKE_CURRENT_LIST_DIR}/src/TestCirLinear.cpp)
target_link_libraries(test_cir_linear PUBLIC LinearLayer Model)

add_executable(test_cir_conv ${CMAKE_CURRENT_LIST_DIR}/src/TestCirConv.cpp)
target_link_libraries(test_cir_conv PUBLIC LinearLayer Model)

add_executable(test_ntt ${CMAKE_CURRENT_LIST_DIR}/src/TestNTT.cpp)
target_link_libraries(test_ntt PUBLIC Utils)

# add_executable(test_tensor ${CMAKE_CURRENT_LIST_DIR}/src/test_tensor.cpp)
# target_link_libraries(test_tensor PUBLIC Datatype)

# add_executable(test_relu ${CMAKE_CURRENT_LIST_DIR}/src/TestReLU2.cpp)
# target_link_libraries(test_relu PUBLIC Model)

add_executable(test_fixpoint ${CMAKE_CURRENT_LIST_DIR}/src/TestFixPoint.cpp)
target_link_libraries(test_fixpoint PUBLIC NonlinearOperator)

# add_executable(test_resnet ${CMAKE_CURRENT_LIST_DIR}/src/TestResNet.cpp)
# target_link_libraries(test_resnet PUBLIC Model)

# add_executable(test_pool ${CMAKE_CURRENT_LIST_DIR}/src/test_Pool.cpp)
# target_link_libraries(test_pool PUBLIC NonlinearLayer)

# add_executable(test_gelu ${CMAKE_CURRENT_LIST_DIR}/src/TestGeLU.cpp)
# target_link_libraries(test_gelu PUBLIC NonlinearLayer)

# add_executable(test_silu ${CMAKE_CURRENT_LIST_DIR}/src/TestSiLU.cpp)
# target_link_libraries(test_silu PUBLIC NonlinearLayer)

# add_executable(test_division ${CMAKE_CURRENT_LIST_DIR}/src/test_division.cpp)
# target_link_libraries(test_division PUBLIC NonlinearOperator)
