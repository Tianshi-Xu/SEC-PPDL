
1. Install Dependencies

     cd Extern/ and follow the instructions in the README.md file.

2. Build the Project

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

    3. Run the tests

    ``` bash
    ./build/Test/test_resnet r=1 p=1234 && ./build/Test/test_resnet r=2 p=1234
    ```
