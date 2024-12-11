#include "Tensor.h"
using namespace std;

// 辅助函数：打印张量的形状
template <typename T, size_t N>
void print_shape(const Tensor<T, N>& tensor) {
    std::vector<size_t> shp = tensor.shape();
    std::cout << "(";
    for (size_t i = 0; i < shp.size(); ++i) {
        std::cout << shp[i];
        if (i != shp.size() - 1) std::cout << ", ";
    }
    std::cout << ")";
}

template <typename T>
void print_tensor_3d(const Tensor<T, 3>& tensor) {
    std::vector<size_t> shp = tensor.shape();
    if (shp.size() != 3) {
        throw std::invalid_argument("print_tensor_3d expects a 3D tensor.");
    }
    size_t dim0 = shp[0];
    size_t dim1 = shp[1];
    size_t dim2 = shp[2];

    for (size_t i = 0; i < dim0; ++i) {
        std::cout << "Slice " << i << ":\n";
        for (size_t j = 0; j < dim1; ++j) {
            for (size_t k = 0; k < dim2; ++k) {
                std::cout << tensor({i, j, k}) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main() {
    try {
        // 定义三维张量的形状
        std::vector<size_t> shape = {2, 3, 4}; // 2 slices, 3 rows, 4 columns

        // 创建两个三维张量
        Tensor<int, 3> tensor1(shape);
        Tensor<int, 3> tensor2(shape);

        // 初始化 tensor1
        // 填充 tensor1 的元素为 i * 100 + j * 10 + k
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    tensor1({i, j, k}) = static_cast<int>(i * 100 + j * 10 + k);
                }
            }
        }

        // 初始化 tensor2
        // 填充 tensor2 的元素为 (i + 1) * 100 + (j + 1) * 10 + (k + 1)
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    tensor2({i, j, k}) = static_cast<int>((i + 1) * 100 + (j + 1) * 10 + (k + 1));
                }
            }
        }

        // 打印 tensor1
        std::cout << "Tensor1 Shape: ";
        print_shape(tensor1);
        std::cout << "\nTensor1 Data:\n";
        print_tensor_3d(tensor1);

        // 打印 tensor2
        std::cout << "Tensor2 Shape: ";
        print_shape(tensor2);
        std::cout << "\nTensor2 Data:\n";
        print_tensor_3d(tensor2);

        // 执行加法
        Tensor<int, 3> tensor_sum = tensor1 + tensor2;
        std::cout << "Sum Tensor Shape: ";
        print_shape(tensor_sum);
        std::cout << "\nSum Tensor Data:\n";
        print_tensor_3d(tensor_sum);

        // 执行减法
        Tensor<int, 3> tensor_diff = tensor1 - tensor2;
        std::cout << "Difference Tensor Shape: ";
        print_shape(tensor_diff);
        std::cout << "\nDifference Tensor Data:\n";
        print_tensor_3d(tensor_diff);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}