cmake_minimum_required(VERSION 3.10)

# file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
# add_library(Utils STATIC ${srcs})
# target_include_directories(Utils PUBLIC include)

# message(STATUS "Utils_FOUND: ${Utils_FOUND}")

message(STATUS "Utils_CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "OPENSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/Utils/cmake)
include(include/Utils/cmake/source_of_randomness.cmake)

# set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl")
# find_package(OpenSSL REQUIRED)
# find_package(GMP REQUIRED)

# add_library(Utils INTERFACE)
# file(GLOB_RECURSE hdrs CONFIGURE_DEPENDS include/*.h)

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})
add_library(Utils INTERFACE)
target_include_directories(Utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(Utils
    INTERFACE
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")
#     "-pthread;-Wall;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-s td=c++17;-w;-g;-ggdb;-O0")

if(USE_RANDOM_DEVICE)
    target_compile_definitions(Utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(Utils INTERFACE "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(Utils
    INTERFACE ${OPENSSL_INCLUDE_DIR} ${GMP_INCLUDE_DIR}
)

target_link_libraries(Utils
    INTERFACE ${OPENSSL_LIBRARIES} ${GMP_LIBRARIES}
)

message(STATUS "Utils_FOUND: ${Utils_FOUND}")