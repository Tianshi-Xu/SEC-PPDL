cmake_minimum_required(VERSION 3.10)

project(PIR)

if(USE_HE_GPU)
    add_library(PIR STATIC
        src/PirMath.cpp
        src/PirRuntime.cpp
        src/PirCiphertextOps.cpp
        src/PirShapeBuilder.cpp
        src/PirDatabaseGenerator.cpp
        src/PirQueryGenerator.cpp
        src/PirAnswerGenerator.cpp
        src/PirGpuExecutor.cpp
        src/BatchPirGpuExecutor.cpp)
    target_include_directories(PIR PUBLIC include)
    target_link_libraries(PIR PUBLIC HE Datatype)
endif()
