## CPU dependencies

Run from the SEC-PPDL repository root with Git, a C++17 compiler, CMake and OpenSSL development headers. Initialize the pinned submodules in a fresh clone, then apply the repository patches once.

```bash
git submodule update --init
git -C Extern/emp-tool apply ../patch/emp-tool.patch
git -C Extern/emp-ot apply ../patch/emp-ot.patch
git -C Extern/SEAL apply ../patch/seal-ckks.patch
cmake -S Extern/hexl -B Extern/hexl/build -DHEXL_TESTING=OFF -DHEXL_BENCHMARK=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/Extern/hexl/hexl_build"
cmake --build Extern/hexl/build -j8
cmake --install Extern/hexl/build
cmake -S Extern/SEAL -B Extern/SEAL/build -DSEAL_USE_INTEL_HEXL=ON \
  -DSEAL_BUILD_DEPS=OFF -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF \
  -DHEXL_DIR="$PWD/Extern/hexl/hexl_build/lib/cmake/hexl-1.2.6"
cmake --build Extern/SEAL/build -j8
```

## GPU dependency

With the CUDA toolkit on your PATH, run `bash Extern/build-phantom.sh` from the repository root. The script checks out the revision in `Extern/phantom-fhe.rev`, applies the bundled patch and builds for the installed GPU. Set `CUDA_ARCHITECTURES` to override the native architecture.
