需要单独build一次SEAL，然后把extern/SEAL/build/native/src/seal/util/config.h复制到extern/SEAL/native/src/seal/util/以解决
fatal error: seal/util/config.h: No such file or directory

    cd src
    cmake -S . -B build
    cmake --build build
    cd build
    ./test_conv 0
    ./test_conv 1
