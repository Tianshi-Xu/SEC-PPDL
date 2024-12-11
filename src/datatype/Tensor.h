#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <numeric>

// 前向声明
template <typename T, size_t N>
class Tensor;

// 特化模板：一维张量
template <typename T>
class Tensor<T, 1> {
public:
    std::vector<T> data;

    // 默认构造函数
    Tensor() {}

    // 使用指定形状初始化张量，初始化为默认值（比如0）
    Tensor(const std::vector<size_t>& shape) {
        resize(shape);
    }

    // 获取张量的形状
    std::vector<size_t> shape() const {
        return { data.size() };
    }

    // 公共的 resize 方法
    void resize(const std::vector<size_t>& shape) {
        if (shape.size() != 1) {
            throw std::invalid_argument("Shape size does not match tensor dimension.");
        }
        data.resize(shape[0], T{}); // 使用默认值初始化
    }

    // 重载 () 运算符 支持访问和赋值
    T& operator()(const std::vector<size_t>& indices) {
        if (indices.size() != 1) {
            throw std::invalid_argument("Incorrect number of indices for 1D tensor.");
        }
        return data[indices[0]];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        if (indices.size() != 1) {
            throw std::invalid_argument("Incorrect number of indices for 1D tensor.");
        }
        return data[indices[0]];
    }

    // 将张量扁平化为一维数组
    std::vector<T> flatten() const {
        return data;
    }

    // 将张量重塑为新的形状
    Tensor<T, 1> reshape(const std::vector<size_t>& new_shape) const {
        if (new_shape.size() != 1) {
            throw std::invalid_argument("New shape does not match tensor dimension.");
        }
        if (new_shape[0] != data.size()) {
            throw std::invalid_argument("New shape must have the same number of elements.");
        }
        Tensor<T, 1> reshaped_tensor(new_shape);
        reshaped_tensor.data = data;
        return reshaped_tensor;
    }

    // 将扁平化数据填充到张量
    void fill_from_flattened(const std::vector<T>& flattened_data, size_t& index) {
        for (auto& val : data) {
            val = flattened_data[index++];
        }
    }

    // 辅助函数，扁平化张量
    void flatten_helper(std::vector<T>& result) const {
        result.insert(result.end(), data.begin(), data.end());
    }

    // 友元函数：加法运算符
    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        // 首先检查两个张量的形状是否相同
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot add Tensors with different shapes.");
        }

        // 创建一个新的 Tensor 以存储结果
        Tensor result(a.shape());
        result.data.resize(a.data.size());

        // 执行元素级加法
        for (size_t i = 0; i < a.data.size(); ++i) {
            result.data[i] = a.data[i] + b.data[i];
        }

        return result;
    }

    // 友元函数：减法运算符
    friend Tensor operator-(const Tensor& a, const Tensor& b) {
        // 首先检查两个张量的形状是否相同
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot subtract Tensors with different shapes.");
        }

        // 创建一个新的 Tensor 以存储结果
        Tensor result(a.shape());
        result.data.resize(a.data.size());

        // 执行元素级减法
        for (size_t i = 0; i < a.data.size(); ++i) {
            result.data[i] = a.data[i] - b.data[i];
        }

        return result;
    }
};

// 一般模板：N维张量（N >= 2）
template <typename T, size_t N>
class Tensor {
public:
    std::vector<Tensor<T, N - 1>> data; // 存储每个元素的数据

    // 默认构造函数：创建一个空的张量
    Tensor() {}

    // 使用指定形状初始化张量，初始化为默认值（比如0）
    Tensor(const std::vector<size_t>& shape) {
        resize(shape);
    }

    // 获取当前张量的形状
    std::vector<size_t> shape() const {
        std::vector<size_t> current_shape;
        current_shape.push_back(data.size());
        if (!data.empty()) {
            auto subshape = data.front().shape();
            current_shape.insert(current_shape.end(), subshape.begin(), subshape.end());
        } else {
            current_shape.insert(current_shape.end(), N - 1, 0);
        }
        return current_shape;
    }

    // 公共的 resize 方法，用于根据给定形状调整张量的大小
    void resize(const std::vector<size_t>& shape) {
        if (shape.size() < 1) {
            throw std::invalid_argument("Shape must have at least one dimension.");
        }
        size_t dim_size = shape[0];
        data.resize(dim_size, Tensor<T, N - 1>()); // 使用默认构造初始化
        std::vector<size_t> sub_shape(shape.begin() + 1, shape.end());
        for (auto& sub_tensor : data) {
            sub_tensor.resize(sub_shape);
        }
    }

    // 重载 () 运算符 支持访问和赋值
    T& operator()(const std::vector<size_t>& indices) {
        if (indices.size() != N) {
            throw std::invalid_argument("Incorrect number of indices.");
        }
        return data[indices[0]](std::vector<size_t>(indices.begin() + 1, indices.end()));
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        if (indices.size() != N) {
            throw std::invalid_argument("Incorrect number of indices.");
        }
        return data[indices[0]](std::vector<size_t>(indices.begin() + 1, indices.end()));
    }

    // 友元函数：加法运算符
    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        // 首先检查两个张量的形状是否相同
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot add Tensors with different shapes.");
        }

        // 创建一个新的 Tensor 以存储结果
        Tensor result(a.shape());
        result.data.resize(a.data.size());

        // 执行元素级加法
        for (size_t i = 0; i < a.data.size(); ++i) {
            result.data[i] = a.data[i] + b.data[i];
        }

        return result;
    }

    // 友元函数：减法运算符
    friend Tensor operator-(const Tensor& a, const Tensor& b) {
        // 首先检查两个张量的形状是否相同
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Cannot subtract Tensors with different shapes.");
        }

        // 创建一个新的 Tensor 以存储结果
        Tensor result(a.shape());
        result.data.resize(a.data.size());

        // 执行元素级减法
        for (size_t i = 0; i < a.data.size(); ++i) {
            result.data[i] = a.data[i] - b.data[i];
        }

        return result;
    }

    // 将张量扁平化为一维数组
    std::vector<T> flatten() const {
        std::vector<T> result;
        flatten_helper(result);
        return result;
    }

    // 将张量重塑为新的形状（需要确保新的形状与原始元素数量一致）
    Tensor<T, N> reshape(const std::vector<size_t>& new_shape) const {
        size_t total_size = 1;
        for (size_t dim : shape()) {
            total_size *= dim;
        }
        size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());

        if (total_size != new_total_size) {
            throw std::invalid_argument("New shape must have the same number of elements.");
        }

        std::vector<T> flattened_data = flatten();
        Tensor<T, N> reshaped_tensor(new_shape);
        size_t index = 0;
        reshaped_tensor.fill_from_flattened(flattened_data, index);
        return reshaped_tensor;
    }

    // 递归的访问和赋值函数
    T& get_element(const std::vector<size_t>& indices) {
        if constexpr (N == 1) {
            return data[indices[0]];
        }
        else {
            return data[indices[0]].get_element(std::vector<size_t>(indices.begin() + 1, indices.end()));
        }
    }

    const T& get_element(const std::vector<size_t>& indices) const {
        if constexpr (N == 1) {
            return data[indices[0]];
        }
        else {
            return data[indices[0]].get_element(std::vector<size_t>(indices.begin() + 1, indices.end()));
        }
    }

    // 将扁平化数据填充到张量
    void fill_from_flattened(const std::vector<T>& flattened_data, size_t& index) {
        for (auto& sub_tensor : data) {
            sub_tensor.fill_from_flattened(flattened_data, index);
        }
    }

    // 辅助函数，扁平化张量
    void flatten_helper(std::vector<T>& result) const {
        for (const auto& sub_tensor : data) {
            sub_tensor.flatten_helper(result);
        }
    }
};
