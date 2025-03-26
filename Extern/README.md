#### SEAL
```
git clone https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DSEAL_USE_INTEL_HEXL=ON
cmake --build build
```
#### HEXL
- HEXL is used for NTT
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
- If you don't use GPU, you can skip this step and set `-DUSE_HE_GPU=ON` when building the project.
```
git clone https://github.com/encryptorion-lab/phantom-fhe.git
cd phantom-fhe
git apply ../patch/phantom.patch
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_INSTALL_PREFIX=build_phantom
cmake --build build --target install --parallel
```
You may need to add the following line to the `~/.bashrc` file and run `source ~/.bashrc`:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
