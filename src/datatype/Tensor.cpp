#include "Tensor.h"
// 示例使用
int main() {
    // 使用double类型的张量
    Tensor<double> tensor_double({2, 3}, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
    // 初始化一个3x3的张量，默认值为0
    Tensor<int> x({3,3});
    x.print();
    
    std::cout << "Double Tensor:\n";
    tensor_double.print();
    tensor_double({0,0})=999;
    // 重塑为3x2
    tensor_double.reshape({3, 2});
    std::cout << "After reshape to [3, 2]:\n";
    tensor_double.print();

    // 展平
    tensor_double.flatten();
    std::cout << "After flatten:\n";
    tensor_double.print();

    // 重新形状为2x3
    tensor_double.reshape({2, 3});
    std::cout << "After reshape back to [2, 3]:\n";
    tensor_double.print();

    // 索引访问
    std::cout << "Element at (1, 2): " << tensor_double({1, 2}) << "\n\n";

    // 使用int32_t类型的张量
    Tensor<int32_t> tensor_int32({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "int32_t Tensor:\n";
    tensor_int32.print();

    // 创建另一个int32_t张量
    Tensor<int32_t> tensor_int32_2({2, 3}, {6, 5, 4, 3, 2, 1});
    tensor_int32_2.print();

    // 加法
    Tensor<int32_t> sum_int32 = tensor_int32 + tensor_int32_2;
    std::cout << "Sum of int32_t tensors:\n";
    sum_int32.print();

    // 减法
    Tensor<int32_t> diff_int32 = tensor_int32 - tensor_int32_2;
    std::cout << "Difference of int32_t tensors:\n";
    diff_int32.print();

    // 使用int64_t类型的张量
    Tensor<int64_t> tensor_int64({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    std::cout << "int64_t Tensor:\n";
    tensor_int64.print();

    // 使用float类型的张量
    Tensor<float> tensor_float({3}, {1.0f, 2.0f, 3.0f});
    std::cout << "float Tensor:\n";
    tensor_float.print();


    return 0;
}
