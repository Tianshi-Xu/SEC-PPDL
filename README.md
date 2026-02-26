# SEC-LAB-PPDL Codebase

An open-source Privacy-Preserving Deep Learning framework by SEC LAB.

- Multiple MPC and HE protocols
- One-click CPU / GPU switching
- One-click VOLE / IKNP OT primitive switching

See [cpp_coding_style.md](cpp_coding_style.md) for C++ coding conventions.

## 1. Install Dependencies

### SEAL

```bash
cd Extern
git clone https://github.com/microsoft/SEAL.git && cd SEAL
git apply ../patch/seal-ckks.patch
cmake -S . -B build -DSEAL_USE_INTEL_HEXL=ON
cmake --build build -j && cd ../..
```

### HEXL

```bash
cd Extern
git clone https://github.com/intel/hexl.git && cd hexl
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=hexl_build
cmake --build build -j && cmake --install build && cd ../..
```

### emp-ot & emp-tool

```bash
cd Extern && . ./build-ot.sh && cd ..
```

### phantom-fhe (GPU, optional)

Skip this if GPU is not needed (use `-DUSE_HE_GPU=OFF` when building).

```bash
cd Extern
git clone https://github.com/encryptorion-lab/phantom-fhe.git && cd phantom-fhe
git apply ../patch/phantom.patch
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_INSTALL_PREFIX=build_phantom
cmake --build build --target install --parallel && cd ../..
```

You may need to add the following to `~/.bashrc` and run `source ~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

## 2. Build

```bash
cmake -S . -B build          # add -DUSE_HE_GPU=ON for GPU support
cmake --build build -j
```

## 3. Run Tests

Tests use a two-party protocol. Start **Server** (`r=1`) before **Client** (`r=2`) with the same port.

### PrivCirNet

`test_cir_linear` and `test_cir_conv` are reference implementations for [PrivCirNet](https://arxiv.org/abs/2407.13418), testing Block Circulant Linear and Convolution layers respectively. More projects will be added.

```bash
# Block Circulant Linear
./build/Test/test_cir_linear r=1 p=1234   # Terminal 1 (Server)
./build/Test/test_cir_linear r=2 p=1234   # Terminal 2 (Client)

# Block Circulant Convolution
./build/Test/test_cir_conv r=1 p=1234     # Terminal 1 (Server)
./build/Test/test_cir_conv r=2 p=1234     # Terminal 2 (Client)
```
