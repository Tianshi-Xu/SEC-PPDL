include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(Phantom SHARED
        context.cu
        batchencoder.cu
        prng.cu
        polymath.cu
        secretkey.cu
        rns.cu
        rns_base.cu
        rns_bconv.cu
        scalingvariant.cu
        evaluate.cu
        fft.cu
        ckks.cu
        galois.cu
        eval_key_switch.cu

        ntt/fntt_2d.cu
        ntt/ntt_1d.cu
        ntt/ntt_keyswitch_old.cu
        ntt/intt_2d.cu
        ntt/ntt_modup.cu
        ntt/ntt_moddown.cu

        host/blake2b.cu
        host/blake2xb.cu
        host/globals.cu
        host/hash.cu
        host/modulus.cu
        host/ntt.cu
        host/numth.cu
        host/rns.cu
        host/uintarith.cu
        host/uintarithmod.cu
        host/uintarithsmallmod.cu
)

target_include_directories(Phantom PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(Phantom PUBLIC rmm::rmm)

if (FP64_MM_ARITH)
    target_compile_definitions(Phantom PUBLIC FP64_MM_ARITH)
endif()

if (RNS_POLY_BATCH)
    target_compile_definitions(Phantom PUBLIC RNS_POLY_BATCH)
endif()
target_compile_options(Phantom PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_compile_options(Phantom PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
set_target_properties(Phantom PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
