# 🔥SEC-LAB-PPDL Codebase 2026

🔥🔥🔥An open-source Privacy-Preserving Deep Learning framework by SEC LAB (Led by Prof. Meng Li).🔥🔥🔥

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

### [PrivCirNet](https://proceedings.neurips.cc/paper_files/paper/2024/file/ca9873918aa72e9033041f76e77b5c15-Paper-Conference.pdf)

`test_cir_linear` and `test_cir_conv` are reference implementations for [PrivCirNet]([https://arxiv.org/abs/2407.13418]([https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca9873918aa72e9033041f76e77b5c15-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/file/ca9873918aa72e9033041f76e77b5c15-Paper-Conference.pdf))), testing Block Circulant Linear and Convolution layers respectively. More projects will be added.

```bash
# Block Circulant Linear
./build/Test/test_cir_linear r=1 p=1234   # Terminal 1 (Server)
./build/Test/test_cir_linear r=2 p=1234   # Terminal 2 (Client)

# Block Circulant Convolution
./build/Test/test_cir_conv r=1 p=1234     # Terminal 1 (Server)
./build/Test/test_cir_conv r=2 p=1234     # Terminal 2 (Client)
```

### FaceResNet64

`Model::FaceResNet64` verifies a pair of aligned RGB face crops with shape
`[3, 64, 64]`. Supply signed Q8 pixels in `[-1, 1]` as additive shares modulo
`2^60`. Use a 60-bit ReLU and BFV plaintext modulus above `2^33`. Both parties
call `Verify` with the same threshold in `(0, 1]`. The method opens one match
bit. `Forward` returns a share of the 128-dimensional embedding.

Convolutions on `64×64` feature maps use Cheetah with partitions selected by
the total number of input and output ciphertexts. Smaller feature maps use
`CirConv2D` with `block_size=1`, including the `1×1` embedding projection.
The single-row `CirLinearNest` path uses the same projection with matrix
weights in input-channel by output-channel order. Use a BFV polynomial
degree of at least 8192 for this model.

The server supplies trained parameters as tensors in OIHW order. Convolution
kernels use signed Q8 and biases use signed Q16. Encode negative integers in
`uint64_t` using two's complement. Weights and biases have real magnitude at
most 4. The tensors go in this order:

| Layer | Input channels | Output channels | Kernel |
| --- | ---: | ---: | ---: |
| Stem | 3 | 16 | 3 |
| Stage 1 convolutions | 16, 16 | 16, 16 | 3, 3 |
| Stage 2 convolutions and shortcut | 16, 32, 16 | 32, 32, 32 | 3, 3, 1 |
| Stage 3 convolutions and shortcut | 32, 64, 32 | 64, 64, 64 | 3, 3, 1 |
| Embedding projection | 64 | 128 | 1 |

The first convolution and shortcut in stages 2 and 3 use stride two. Hidden
activations are clipped to `[0, 8]` and embeddings to `[-1, 1]`. Calibrate the
threshold on quantized embeddings. The protocol rounds the squared threshold
up to a multiple of `1/4096` and rejects zero embeddings.

#### Run the baseline and ours

Build the CPU test from the repository root after installing the dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HE_GPU=OFF
cmake --build build --target test_face_resnet -j
```

Start each server and client in separate terminals. Use the same port and
flags on both sides. The benchmark uses identical synthetic weights and
crops for both configurations.

```bash
# Baseline: all convolutions use the original Cheetah partition strategy.
# Ring/field conversion and truncation are exact.
./build/Test/test_face_resnet r=1 p=1234 full=1 baseline=1 mm=0
./build/Test/test_face_resnet r=2 p=1234 full=1 baseline=1 mm=0

# Ours: mixed Cheetah/CirConv with M&M conversion and fused truncation.
./build/Test/test_face_resnet r=1 p=1240 full=1 baseline=0 mm=1
./build/Test/test_face_resnet r=2 p=1240 full=1 baseline=0 mm=1
```

Use `baseline=0 mm=0` on both sides to measure the mixed convolution
backend with exact conversion and truncation. Omit `full=1` to run the
operator checks alone.

Each process prints its own bytes sent across two verification pairs. For
each metric, add the two parties' byte counts and divide by `2 × 2^20` to
obtain bidirectional MiB per pair. Each pair runs two model forwards. The
measurement starts after setup and the operator checks. Convolution
communication counts the ciphertext exchange. Linear-layer communication
also includes modulus conversion and truncation. Verification communication
includes the full forward and secure match.

#### Communication results

CPU loopback benchmark with IKNP, three OT threads, BFV degree 8192, and a
50-bit plaintext modulus. All values are bidirectional MiB per verification
pair.

| Configuration | Convolution ciphertexts | Conversion and truncation | Linear layers total | End-to-end verification |
| --- | ---: | ---: | ---: | ---: |
| Cheetah baseline (`baseline=1 mm=0`) | 1403.32 | 1648.67 | 3051.99 | 4287.04 |
| Mixed convolutions, exact conversion (`baseline=0 mm=0`) | 389.09 | 1648.67 | 2037.76 | 3272.81 |
| Ours (`baseline=0 mm=1`) | 389.09 | 36.59 | 425.68 | 1660.73 |
| Ours reduction from baseline | 72.3% | 97.8% | 86.1% | 61.3% |

#### Enable M&M in an application

Pass `use_mm_conversion=true` to the `FaceResNet64` constructor on both
parties to enable M&M-style bounded modulus conversion. This path uses
one OT per input value and fuses field-to-ring conversion with truncation
using one correlated OT per output value. Fused truncation can differ from
exact truncation by one Q8 unit per convolution. The default path uses
exact truncation.

## Citation

If you use this codebase, please cite both of the following:

```bibtex
@article{xu2024privcirnet,
  title={Privcirnet: Efficient private inference via block circulant transformation},
  author={Xu, Tianshi and Wu, Lemeng and Wang, Runsheng and Li, Meng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={111802--111831},
  year={2024}
}

@inproceedings{xu2025breaking,
  title={Breaking the layer barrier: Remodeling private transformer inference with hybrid {CKKS} and {MPC}},
  author={Xu, Tianshi and Lu, Wen-jie and Yu, Jiangrui and Chen, Yi and Lin, Chenqi and Wang, Runsheng and Li, Meng},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={2653--2672},
  year={2025}
}
```
