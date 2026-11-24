# 🔥SEC-LAB-PPDL Codebase 2026

🔥🔥🔥An open-source Privacy-Preserving Deep Learning framework by SEC LAB (Led by Prof. Meng Li).🔥🔥🔥

- Multiple MPC and HE protocols
- One-click CPU / GPU switching
- One-click VOLE / IKNP OT primitive switching

See [cpp_coding_style.md](cpp_coding_style.md) for C++ coding conventions.

## OptiPrime

OptiPrime (MICRO 2026) introduces OptiEncode, the encrypted convolution protocol implemented here for CPU and GPU. [Paper](https://arxiv.org/abs/2609.16898) · [BibTeX](#cite-optiprime)

For two-party quantized inference with ResNet-18/34/50 or VGG-16/19, see the [experiment guide](Experiments/OptiPrime/README.md) and [network definitions](Experiments/OptiPrime/imagenet/architecture.py).

### OptiEncode convolution

Link the CMake target `LinearLayer` and use `LinearLayer::Conv2DOptiEncode` with the host or GPU HE backend. Both parties supply additive input shares. The server supplies OIHW weights and per-channel bias as residues of the plaintext modulus. The client passes zero tensors with the same shapes. For a power-of-two plaintext modulus, set `options.ciphertext_inverse_normalization = true` on both parties. [Public header](Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h)

```cpp
#include <LinearLayer/ConvOptiEncode.h>

he.GenerateNewKey(LinearLayer::Conv2DOptiEncode::GaloisElements(he.polyModulusDegree));
LinearLayer::OptiEncodeOptions options;
options.threads = 4;
LinearLayer::Conv2DOptiEncode conv(height, stride, padding, weight, bias, &he, options);
auto output_share = conv(input_share);
```

For HE setup and input sharing, see the [two-party operator test](Test/src/TestOptiEncodeConv.cpp). [Run the OptiEncode tests](#optiencode-tests)

## Citation

Please cite PrivCirNet and Breaking the Layer Barrier when using SEC-PPDL. If your work uses the OptiEncode convolution protocol, please also cite [OptiPrime (MICRO 2026)](#cite-optiprime).

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

### Cite OptiPrime

```bibtex
@inproceedings{yu2026optiprime,
  title={{OptiPrime}: Optimizing Private Inference through Protocol-Hardware Co-design},
  author={Yu, Jiangrui and Yu, Ye and Chen, Si and Lin, Chenqi and Zeng, Wenxuan and Fan, Junfeng and Gao, Mingyu and Li, Meng},
  booktitle={Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2026},
  url={https://arxiv.org/abs/2609.16898}
}
```

## 1. Install Dependencies

Build the pinned CPU dependencies and, for GPU support, Phantom. [Dependency instructions](Extern/README.md)

## 2. Build

```bash
cmake -S . -B build          # add -DUSE_HE_GPU=ON for GPU support
cmake --build build -j
```

## 3. Run Tests

Tests use a two-party protocol. Start **Server** (`r=1`) before **Client** (`r=2`) with the same port.

### OptiEncode tests

From the repository root, run the CPU suite after building with `BUILD_TESTING=ON` (the default). It checks the encoding algebra, two-party convolution against a clear reference, default Galois keys, and FaceResNet integration. CTest starts both parties automatically.

```bash
ctest --test-dir build --output-on-failure -R '^optiencode_'
```

For the GPU convolution check, first install the [GPU dependency](Extern/README.md#gpu-dependency), then run:

```bash
cmake -S . -B build-gpu -DUSE_HE_GPU=ON -DBUILD_TESTING=ON
cmake --build build-gpu --target test_optiencode_conv -j8
ctest --test-dir build-gpu --output-on-failure -R '^optiencode_gpu_conv$'
```

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
`[3, 64, 64]` in CHW order. Supply pixels in `[-1, 1]` as Q12 additive shares
modulo `2^37`. The model uses `ell=37` and `k=12` from the
[OpenCheetah release scripts](https://github.com/Alibaba-Gemini-Lab/OpenCheetah/blob/f71d4d49ffde81701fc3cc2d887fae62381fb472/scripts/common.sh#L51-L54).
Set the secure ReLU ring to 37 bits and use a BFV plaintext modulus above
`2^33`.

| Backbone | Stem | Residual stage depths | Stage widths | Map before global average pooling |
| --- | --- | --- | --- | --- |
| `Backbone::ResNet18` | 7×7 convolution, stride 2, then 3×3 MaxPool, stride 2 | 2, 2, 2, 2 | 64, 128, 256, 512 | 2×2 |
| `Backbone::ResNet20` | 3×3 convolution, stride 1 | 3, 3, 3 | 16, 32, 64 | 16×16 |
| `Backbone::ResNet32` | 3×3 convolution, stride 1 | 5, 5, 5 | 16, 32, 64 | 16×16 |
| `Backbone::ResNet34` | 7×7 convolution, stride 2, then 3×3 MaxPool, stride 2 | 3, 4, 6, 3 | 64, 128, 256, 512 | 2×2 |
| `Backbone::Compact` | 3×3 convolution, stride 1 | 1, 1, 1 | 16, 32, 64 | 16×16 |

The ResNet backbones use basic residual blocks and ordinary ReLU. The first
block of each stage after stage 1 downsamples with a stride-two convolution
and a 1×1 projection shortcut. ResNet20 and ResNet32 use the CIFAR-style stem
and three stages. After global average pooling, a learned projection
replaces the classification head and produces a 128-dimensional embedding.
`Forward` returns an embedding share with coordinates clipped to `[-1, 1]`.
Both parties call `Verify` with the same threshold in `(0, 1]` and receive one
match bit. The cosine comparison widens its three shared dot products to
64 bits for squaring, rounds the squared threshold up to a multiple of
`1/4096`, and rejects zero embeddings. Calibrate the threshold with the
encoded model and its embedding head.

Convolutions on `64×64` feature maps use Cheetah with partitions selected by
the total number of input and output ciphertexts. Smaller feature maps use
`CirConv2D` with `block_size=1`, including the `1×1` embedding projection.
The single-row `CirLinearNest` path uses the same projection with matrix
weights in input-channel by output-channel order. Use a BFV polynomial
degree of at least 8192 for this model.

On the server, fold any evaluation-mode BatchNorm into the preceding
convolution and pass the resulting FP32 tensors to `EncodeWeights`.
It encodes kernels as Q12 and biases as Q24 for secure inference. Use
`ConvolutionShapes(backbone)` for the OIHW shapes and ordering. The order is
stem, each residual block's two main convolutions followed by its shortcut
if present, and the embedding projection. Pass the same backbone to both
constructors. Q24 convolution accumulators must fit the signed 37-bit ring
and have magnitude below half the BFV modulus, or a quarter with M&M.

#### Run the baseline and ours

Build the CPU test from the repository root after installing the dependencies.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HE_GPU=OFF
cmake --build build --target test_face_resnet -j
```

To run both parties on one host with Python 3.11 or later, use
`Test/run_face_resnet.py`. Select the backbone with `--backbone 18`,
`--backbone 20`, `--backbone 32`, or `--backbone 34`.

```bash
# Measure and compare the two protocols over loopback.
python3 Test/run_face_resnet.py --backbone 18 --protocol both

# Measure the hybrid protocol at 400 Mb/s per direction and 20 ms RTT.
python3 Test/run_face_resnet.py --backbone 34 --protocol hybrid --wan
```

The script uses dense FP32 fixtures, Ferret VOLE, and 16 HE workers in
hybrid mode. It saves each party's log and a `summary.json` with latency and
communication breakdowns in a fresh run directory. Use `--binary` for a
non-default build, `--output-dir` to choose where runs are saved,
and `--server-cpus` and `--client-cpus` to pin each party with Linux
`taskset`. The WAN proxy shares the configured bandwidth per direction
across all three connections. Use a different `--port` for concurrent runs.

For separate hosts or manual setup, start each server and client in
separate terminals.

Use the same port and flags on both sides. Select ResNet18, ResNet20, ResNet32, or ResNet34 with
`backbone=18`, `backbone=20`, `backbone=32`, or `backbone=34`. The default
`backbone=0` selects the compact model. Add `dense=1` on both sides to use
synthetic FP32 weights with every convolution and projection coefficient
nonzero after Q12 encoding. The benchmark uses identical weights and crops
for both protocol configurations.

```bash
# Baseline: all convolutions use the original Cheetah partition strategy.
# Ring/field conversion and truncation are exact.
./build/Test/test_face_resnet r=1 p=1234 full=1 backbone=18 baseline=1 mm=0
./build/Test/test_face_resnet r=2 p=1234 full=1 backbone=18 baseline=1 mm=0

# Ours: mixed Cheetah/CirConv with M&M conversion and fused truncation.
./build/Test/test_face_resnet r=1 p=1240 full=1 backbone=18 baseline=0 mm=1
./build/Test/test_face_resnet r=2 p=1240 full=1 backbone=18 baseline=0 mm=1
```

Use `baseline=0 mm=0` on both sides to measure the mixed convolution
backend with exact conversion and truncation. Omit `full=1` to run the
operator checks alone.

#### Run with VOLE-style OT

Use `vole=1` to select Ferret. Set `checks=0` for benchmarking so the
standalone operator checks do not consume preprocessed pairs before the
measured workload. The full test still verifies a matching pair and a
nonmatching pair. Use a fresh working directory for each two-party session
when measuring cold-start setup.

```bash
# Cheetah baseline
./build/Test/test_face_resnet r=1 p=1234 full=1 backbone=18 baseline=1 mm=0 vole=1 checks=0
./build/Test/test_face_resnet r=2 p=1234 full=1 backbone=18 baseline=1 mm=0 vole=1 checks=0

# Ours
./build/Test/test_face_resnet r=1 p=1240 full=1 backbone=18 baseline=0 mm=1 vole=1 checks=0
./build/Test/test_face_resnet r=2 p=1240 full=1 backbone=18 baseline=0 mm=1 vole=1 checks=0
```

Each process prints its own bytes sent across two verification pairs. For
each metric, add the two parties' byte counts and divide by `2 × 2^20` to
obtain bidirectional MiB per pair. Each pair runs two model forwards. The
measurement starts after setup and the operator checks. Convolution
communication counts the ciphertext exchange. Linear-layer communication
also includes modulus conversion and truncation. Verification communication
includes the full forward and secure match.

Each party reports session setup, local model packing, and the duration of
each verification separately. Use the slower party's time for each pair.
The verification timer includes both forwards, secure matching, and any
OT refill triggered during the request. The test also reports the average
model forward time across the four images in the two requests.

For Ferret, the test counts base setup and initial pair generation once per
session. It also counts any random COT generation triggered during
verification as OT refill traffic. Online communication counts the
choice adjustments, OT payloads, and ciphertext exchanges. The setup
and refill counters are separate from the online counters.

#### VOLE communication with 37-bit shares and Q12

The CPU loopback runs use three OT threads, BFV degree 8192, and a 50-bit
plaintext modulus. Each pair uses two forwards and one private cosine
comparison. The table reports bidirectional MiB per verification pair,
averaged over two pairs. The baseline uses this repository's Cheetah
convolution path with input-ciphertext partition cost and full output
ciphertext serialization. The upstream OpenCheetah implementation also
[truncates the output ciphertext and removes unused coefficients](https://github.com/Alibaba-Gemini-Lab/OpenCheetah/blob/f71d4d49ffde81701fc3cc2d887fae62381fb472/include/gemini/cheetah/hom_conv2d_ss.cc#L696-L711).

| Backbone and protocol | Convolution ciphertexts | Online conversion and truncation | Other online verification | Online total | Offline OT refill |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18, Cheetah baseline | 6706.55 | 102.72 | 45.09 | 6854.36 | 5.80 |
| ResNet18, ours | 386.09 | 12.93 | 45.09 | 444.12 | 1.58 |
| ResNet32, Cheetah baseline | 3595.83 | 675.29 | 169.79 | 4440.91 | 34.81 |
| ResNet32, ours | 1413.33 | 92.40 | 169.79 | 1675.52 | 7.38 |

Other online verification includes ReLU, pooling, embedding clipping, and
cosine matching. Each cold session exchanges another 8.61 MiB for Ferret
setup and initial pair generation and 60.70 MiB for HE key exchange.

The convolution ciphertext counts below are per model forward.

| Backbone and protocol | Input ciphertexts | Output ciphertexts |
| --- | ---: | ---: |
| ResNet18, Cheetah baseline | 41 | 6664 |
| ResNet18, ours | 145 | 241 |
| ResNet32, Cheetah baseline | 202 | 3393 |
| ResNet32, ours | 660 | 753 |

In the Cheetah baseline, the last two ResNet18 stages have 256 and 512 output
channels. Their partitions, including projection shortcuts, account for
5120 output ciphertexts per forward. In the hybrid path, a 3×3 CirConv on a
32×32 feature map in ResNet32 rounds the padded spatial size from 34 to 64.
With BFV degree 8192 and block size 1, one ciphertext carries one channel.
The same configuration carries 16 channels when the input
map is 8×8, as in ResNet18.

#### ResNet18 and ResNet34 latency

The runs used 64×64 inputs, Ferret, and three OT threads
per party on an AMD EPYC 9K65 host. Each row averages two requests and uses
the slower party's time for each request. The WAN proxy applied a shared
400 Mb/s limit in each direction and 10 ms one-way delay across the three
protocol connections.

| Backbone | Protocol and request | Synthetic weights | CPU loopback | Simulated WAN |
| --- | --- | --- | ---: | ---: |
| ResNet18 | Cheetah baseline, verification pair | Sparse | 79.0 s | — |
| ResNet18 | Hybrid with M&M, verification pair | Sparse | 16.0 s | 39.7 s |
| ResNet18 | Hybrid with M&M, one secure embedding | Sparse | — | 20.3 s |
| ResNet34 | Hybrid with M&M, verification pair | Sparse | 24.3 s | 63.3 s |
| ResNet34 | Hybrid with M&M, one secure embedding | Sparse | — | 31.7 s |
| ResNet34 | Hybrid with M&M, one secure embedding | Dense | — | 95.6 s |

WAN session setup and server-side model packing took 4.6 s and 2.8 s for
ResNet18 and 5.5 s and 5.0 s for ResNet34 before timed pair verification.
Pair verification runs two model forwards and the private match. The
single-image measurement times one model forward through the embedding head.
The dense latency fixture fills the otherwise-zero kernels with alternating
Q12 values of ±1 and keeps the sparse fixture's nonzero coefficients. The
dense and sparse forward runs sent the same number of bytes in each direction.

#### Dense-weight WAN comparison

All four backbones used 64×64 inputs, `full=1 dense=1 vole=1 checks=0 baseline=0 mm=1`,
and the same 400 Mb/s per direction, 20 ms RTT WAN proxy. Feature maps at
64×64 used Cheetah. Every CirConv layer used `block_size=1`. Setup and weight
packing are outside the timed requests.

| Backbone | Forward per image | End-to-end per verification pair | Online MiB per pair | Offline OT refill MiB per pair |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 56.6 s | 114.2 s | 444.12 | 1.58 |
| ResNet20 (CIFAR-style) | 58.7 s | 118.4 s | 1066.69 | 4.22 |
| ResNet32 (CIFAR-style) | 90.3 s | 181.6 s | 1675.52 | 7.38 |
| ResNet34 | 96.3 s | 193.5 s | 665.78 | 2.37 |

Each run completed two verification requests with synthetic FP32 weights,
one matching and one nonmatching. Forward times average the four sequential
model calls. Pair times include both forwards, private matching, and any
OT refill triggered by the request. Byte totals are bidirectional.

CPU inference in the hybrid mode defaults to `he_threads=16`. This enables
parallel Cheetah output tiles and CirConv input rotations and giant-step
groups. Pass `he_threads=1` to reproduce the single-worker hybrid timings.
The Cheetah baseline uses a single HE worker. Both modes use three OT
threads per party. On the same WAN workload, with 16 physical cores
assigned to the server on one socket, all four backbones came in below
50 seconds per forward.

| Backbone | Forward per image, 1 HE worker | Forward per image, 16 HE workers | Verification pair, 16 HE workers |
| --- | ---: | ---: | ---: |
| ResNet18 | 56.6 s | 19.9 s | 40.8 s |
| ResNet20 (CIFAR-style) | 58.7 s | 30.9 s | 62.8 s |
| ResNet32 (CIFAR-style) | 90.3 s | 49.2 s | 99.4 s |
| ResNet34 | 96.3 s | 33.5 s | 68.0 s |

Both configurations exchanged identical byte totals in each direction.

#### Dense-weight Cheetah baseline comparison

The baseline uses `full=1 dense=1 vole=1 checks=0 baseline=1 mm=0`.
It uses the repository's Cheetah partitions and full output ciphertexts.
The baseline was measured over CPU loopback. The hybrid rows use the WAN
runs above. Communication is bidirectional MiB per verification pair.

| Backbone | Protocol | Convolution ciphertexts | Online conversion and truncation | Other online verification | Online total | Offline OT refill |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | Cheetah baseline | 6706.55 | 102.72 | 45.09 | 6854.36 | 5.80 |
| ResNet18 | Hybrid | 386.09 | 12.93 | 45.09 | 444.12 | 1.58 |
| ResNet20 | Cheetah baseline | 2499.58 | 432.85 | 105.67 | 3038.10 | 22.15 |
| ResNet20 | Hybrid | 901.21 | 59.81 | 105.67 | 1066.69 | 4.22 |
| ResNet32 | Cheetah baseline | 3595.83 | 675.29 | 169.79 | 4440.91 | 34.81 |
| ResNet32 | Hybrid | 1413.33 | 92.40 | 169.79 | 1675.52 | 7.38 |
| ResNet34 | Cheetah baseline | 13012.00 | 156.91 | 59.43 | 13228.34 | 8.97 |
| ResNet34 | Hybrid | 586.14 | 20.21 | 59.43 | 665.78 | 2.37 |

End-to-end online communication improves by 15.43×, 2.85×, 2.65×,
and 19.87× for ResNet18, ResNet20, ResNet32, and ResNet34, respectively.
Each cold session also exchanges 8.61 MiB for Ferret initialization and
60.70 MiB for HE key exchange in either configuration.

Cheetah baseline loopback latency averaged 106.8, 146.0, 228.1, and 190.2
seconds per verification pair for those four backbones, respectively.

#### Historical compact-backbone communication

The tables below are historical measurements for the compact backbone with
60-bit shares, Q8 precision, and clipped hidden activations. The CPU loopback
benchmarks used three OT threads, BFV degree 8192, and a 50-bit plaintext
modulus. The IKNP table reports bidirectional MiB per verification pair.

| Configuration | Convolution ciphertexts | Conversion and truncation | Linear layers total | End-to-end verification |
| --- | ---: | ---: | ---: | ---: |
| Cheetah baseline (`baseline=1 mm=0`) | 1403.32 | 1648.67 | 3051.99 | 4287.04 |
| Mixed convolutions, exact conversion (`baseline=0 mm=0`) | 389.09 | 1648.67 | 2037.76 | 3272.81 |
| Ours (`baseline=0 mm=1`) | 389.09 | 36.59 | 425.68 | 1660.73 |
| Ours reduction from baseline | 72.3% | 97.8% | 86.1% | 61.3% |

With Ferret VOLE-style OT, setup is reported once per cold session. All other
rows are bidirectional MiB per verification pair, averaged across two pairs.

| Measurement | Cheetah baseline | Ours |
| --- | ---: | ---: |
| Ferret OT setup and initial pair generation per session | 8.61 | 8.61 |
| Other setup per session (HE key exchange) | 60.70 | 60.70 |
| Offline OT refill during verification | 13.71 | 5.54 |
| Online convolution ciphertexts | 1403.32 | 389.09 |
| Online linear layers including conversion and truncation | 1610.98 | 418.27 |
| Online end-to-end verification | 1751.75 | 559.04 |
| Verification including OT refill | 1765.46 | 564.58 |

Ours reduces online linear-layer communication by 74.0% and online
end-to-end verification communication by 68.1% relative to the VOLE baseline.

#### Enable M&M in an application

Pass `use_mm_conversion=true` to the `FaceResNet64` constructor on both
parties to enable M&M-style bounded modulus conversion. This path uses
one OT per input value and fuses field-to-ring conversion with truncation
using one correlated OT per output value. Fused truncation can differ from
exact truncation by one Q12 unit per convolution. The default path uses
exact truncation.
