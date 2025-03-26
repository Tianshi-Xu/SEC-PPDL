cmake_minimum_required(VERSION 3.10)
# project(Datatype_test LANGUAGES CXX)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(NonlinearLayer STATIC ${srcs})
target_include_directories(NonlinearLayer PUBLIC include)

target_link_libraries(NonlinearLayer PUBLIC
    HE
    NonlinearOperator
    LinearOperator
)
message(STATUS "Nonlinear_FOUND: ${Nonlinear_FOUND}")