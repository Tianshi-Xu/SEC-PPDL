#### SEAL
```
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git apply ../patch/seal-ckks.patch
cmake -S . -B build -DSEAL_USE_INTEL_HEXL=ON
cmake --build build -j
```
#### HEXL
- HEXL is used for NTT
```
git clone https://github.com/intel/hexl.git
cd hexl
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=hexl_build
cmake --build build -j
cmake --install build
```
#### emp-ot & emp-tool
```
. ./build-ot.sh
```
