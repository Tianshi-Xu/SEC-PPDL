#### SEAL
```
git clone https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DSEAL_USE_INTEL_HEXL=ON
cmake --build build
```
#### HEXL
```
git clone https://github.com/intel/hexl.git
cd hexl
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=hexl_build
cmake --build build
cmake --install build
```
#### emp-ot & emp-tool
```
. ./build-ot.sh
```
#### phantom-fhe
```
git clone https://github.com/encryptorion-lab/phantom-fhe.git
cd phantom-fhe
git apply ../patch/phantom.patch
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_INSTALL_PREFIX=build_phantom
cmake --build build --target install --parallel
```