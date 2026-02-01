cmake_minimum_required(VERSION 3.10)


project(Utils_test)

# HEXL for NTT
find_package(HEXL 1.2.6 REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../Extern/hexl/hexl_build" NO_DEFAULT_PATH)
message(STATUS "HEXL_FOUND for Utils: ${HEXL_FOUND}")

# EMP-TOOL相关配置
set(EMP-TOOL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Extern/emp-tool")
set(EMP-TOOL_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../Extern/emp-tool")
set(EMP-OT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Extern/emp-ot")

# 设置全局include目录，供其他模块使用
include_directories(${EMP-OT_INCLUDE_DIR} ${EMP-TOOL_INCLUDE_DIR})

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(Utils STATIC ${srcs})
target_compile_options(Utils
    PUBLIC
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")
if(USE_RANDOM_DEVICE)
    target_compile_definitions(Utils PUBLIC EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(Utils PUBLIC "-mrdseed")
endif(USE_RANDOM_DEVICE)
target_include_directories(Utils PUBLIC include)

# target_include_directories(Utils
#     PUBLIC ${OPENSSL_INCLUDE_DIR} ${GMP_INCLUDE_DIR}
# )

target_link_libraries(Utils
    PUBLIC ${OPENSSL_LIBRARIES} ${GMP_LIBRARIES}
    HEXL::hexl
)


add_executable(Utils_test ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(Utils_test Utils)


message(STATUS "Utils_FOUND: ${Utils_FOUND}")