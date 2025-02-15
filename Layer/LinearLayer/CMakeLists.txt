project(LinearLayer_test)

find_package(HEXL 1.2.5 REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../Extern/hexl/hexl_build" NO_DEFAULT_PATH)
message(STATUS "HEXL_FOUND: ${HEXL_FOUND}")


file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(LinearLayer STATIC ${srcs})
target_include_directories(LinearLayer PUBLIC include)

target_link_libraries(LinearLayer PUBLIC
    Operator HEXL::hexl
)

message(STATUS "LinearLayer_FOUND: ${LinearLayer_FOUND}")