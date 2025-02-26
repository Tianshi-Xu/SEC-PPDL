cmake_minimum_required(VERSION 3.10)
project(OTPrimitive LANGUAGES CXX)


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(OTPrimitive STATIC ${srcs})
target_include_directories(OTPrimitive PUBLIC include)
message(STATUS "EMP-OT_INCLUDE_DIRS ${EMP-OT_INCLUDE_DIRS}")
message(STATUS "EMP-TOOL_INCLUDE_DIRS ${EMP-TOOL_INCLUDE_DIRS}")
# target_include_directories(OTPrimitive PUBLIC 
#     ${EMP-OT_INCLUDE_DIRS}
#     ${EMP-TOOL_INCLUDE_DIRS}
# )

# target_include_directories(OTPrimitive PUBLIC emp-ot emp-tool)

target_link_libraries(OTPrimitive PUBLIC
    Datatype Utils
)

message(STATUS "OTPrimitive_BUILD_SUCCESS")