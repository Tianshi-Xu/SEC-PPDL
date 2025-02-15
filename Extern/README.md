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
cmake -S . -B build
cmake --build build
```