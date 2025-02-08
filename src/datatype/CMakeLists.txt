find_package(SEAL 4.1.2)

add_library(LINEAR
    BatchNorm.cpp
    Conv.cpp
    Linear.cpp
)

target_link_libraries(LINEAR
    SEAL::seal
)