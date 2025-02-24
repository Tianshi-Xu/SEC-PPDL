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
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=hexl_build
cmake --build build
cmake --install build
```
#### emp-ot & emp-tool
```
cd Extern
. ./build-ot.sh
```