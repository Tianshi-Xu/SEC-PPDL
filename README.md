
1. Pull Submodules

     Run the following command in the project root directory:

    ``` bash
    git submodule init && git submodule update
    ```

2. Apply Patch to the Phantom
   
   This project depends on *Phantom*, and a patch needs to be applied to it. Follow these steps:

   ``` bash
   cd Extern/phantom # Navigate to the root directory of the Phantom
   git apply ../0001-more-flexible.patch
   ```

3. Build the Project

    ``` bash
    cmake -S . -B build
    ```

    **Note**: If GPU support is required, add the `-DUSE_HE_GPU=ON` option to the above command:

    ``` bash
    cmake -S . -B build -DUSE_HE_GPU=ON
    ```

    Compile the project.

    ``` bash
    cmake --build build -j
    ```
    The `-j` option is used for parallel compilation. You can adjust the number of parallel tasks based on your CPU cores.
