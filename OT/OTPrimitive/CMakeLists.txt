cmake_minimum_required(VERSION 3.10)
project(OTPrimitive LANGUAGES CXX)


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(OTPrimitive STATIC ${srcs})
target_include_directories(OTPrimitive PUBLIC include)

target_link_libraries(OTPrimitive PUBLIC
    Datatype Utils
)

message(STATUS "OTPrimitive_BUILD_SUCCESS")