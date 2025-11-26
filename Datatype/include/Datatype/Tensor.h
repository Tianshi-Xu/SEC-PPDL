#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <string>
#include <algorithm>
#include "Datatype/UnifiedType.h"

namespace Datatype {
enum OT_TYPE { IKNP = 0, VOLE };
enum CONV_TYPE { Cheetah = 0, Nest};
enum PARTY {SERVER = 1, CLIENT = 2 };
// Tensor类定义
template <typename T>
class Tensor {
public:
    // for fixed point number
    int32_t bitwidth = 0;
    int32_t scale = 0;
    // 构造函数
    Tensor() = default;
    
    explicit Tensor(const std::vector<size_t>& shape)
        : shape_(shape) {
        computeStrides();
        data_.resize(totalSize(), T(0));
        computeStrides();
    }

    explicit Tensor(const std::vector<size_t>& shape, LOCATION loc)
        : shape_(shape) {
        computeStrides();
        data_.resize(totalSize(), T(loc));
        computeStrides();
    }

    explicit Tensor(const std::vector<size_t>& shape, int32_t bitwidth, int32_t scale)
        : shape_(shape), bitwidth(bitwidth), scale(scale) {
        computeStrides();
        data_.resize(totalSize(), T(0));
        computeStrides();
    }

    explicit Tensor(const std::vector<size_t>& shape, LOCATION loc, int32_t bitwidth, int32_t scale)
        : shape_(shape), bitwidth(bitwidth), scale(scale) {
        computeStrides();
        data_.resize(totalSize(), T(loc));
        computeStrides();
    }


    Tensor(const std::vector<size_t>& shape, T zeros_ct)
        : shape_(shape) {
        computeStrides();
        data_.resize(totalSize(), zeros_ct);
        computeStrides();
    }

    // 从初始化列表构造
    Tensor(const std::vector<size_t>& shape, const std::initializer_list<T>& values)
        : shape_(shape), data_(values) {
        // static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
        computeStrides();
        assert(data_.size() == totalSize() && "Data size does not match shape");
    }

    // Randomize a tensor mod Q, must be integer type. Can not be applied in secure application!
    void randomize(uint64_t Q) {
        // change seed randomly
        srand(time(0));
        if constexpr (std::is_signed_v<T>) {
            // 对于整型：使用按位与实现模运算
            for (size_t i = 0; i < data_.size(); ++i) {
                data_[i] = static_cast<T>(rand()) % Q;
                data_[i] -= static_cast<T>(Q/2);
            }
        }
        else if constexpr (std::is_unsigned_v<T>) {
            for (size_t i = 0; i < data_.size(); ++i) {
                data_[i] = static_cast<T>(rand()) % Q;
            }
        }
        else{
            std::cerr << "Randomize for non-integer type under modulous is not supported" << std::endl;
        }
    }

    // Randomize a tensor without mod, can be any type like float
    void randomize(){
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<T>(rand()/ double(RAND_MAX));
        }
    }
    
    // Flatten操作
    void flatten() {
        shape_ = { data_.size() };
        strides_ = { 1 };
    }
    // Reshape操作
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
        assert(new_size == data_.size() && "Total size must remain unchanged in reshape");
        shape_ = new_shape;
        computeStrides();
    }

    const T& operator()(const size_t& indice) const {
        return data_[indice];
    }  

    T& operator()(const size_t& indice) {
        return data_[indice];
    }    

    // 索引访问（通过向量索引）
    T& operator()(const std::vector<size_t>& indices) {
        size_t idx = computeIndex(indices);
        return data_[idx];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        size_t idx = computeIndex(indices);
        return data_[idx];
    }

    // 索引访问（通过初始化列表）
    T& operator()(std::initializer_list<size_t> indices) {
        return operator_(std::vector<size_t>(indices));
    }

    const T& operator()(std::initializer_list<size_t> indices) const {
        return operator_(std::vector<size_t>(indices));
    }

    // 获取形状
    const std::vector<size_t>& shape() const { return shape_; }
    
    const size_t size() const { return totalSize(); }

    // 获取数据
    const std::vector<T>& data() const { return data_; }

    std::vector<T>& data(){ return data_; }

    // Apply Element-wise function
    template <typename Func>
    void apply(Func func) {
        for (auto& element : data_) {
            func(element);
        }
    }

    // 加法
    Tensor<T> operator+(const Tensor<T>& other) const {
        static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
        assert(shape_ == other.shape_ && "Shapes must match for addition");
        //TODO only support arithmetic types
        Tensor<T> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    // 减法
    Tensor<T> operator-(const Tensor<T>& other) const {
        static_assert(std::is_arithmetic<T>::value, "Tensor only supports arithmetic types.");
        assert(shape_ == other.shape_ && "Shapes must match for subtraction");
        Tensor<T> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    // multiply a scalar
    Tensor<T> operator*(const T& scalar) const {
        Tensor<T> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    void print_shape() const {
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i != shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    // 打印张量（仅支持打印数据和形状）
    void print(size_t num_print = 1e7) const {
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i != shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\nData: [";
        size_t size = std::min(this->size(), num_print);
        for (size_t i = 0; i < size; ++i) {
            if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
                std::cout << static_cast<int>(data_[i]);
            } else if constexpr (std::is_same_v<T, __int128> || std::is_same_v<T, __int128_t>) {
                std::cout << to_string_int128(static_cast<__int128>(data_[i]));
            } else if constexpr (std::is_same_v<T, __uint128_t>) {
                std::cout << to_string_uint128(static_cast<__uint128_t>(data_[i]));
            } else {
                std::cout << data_[i];
            }
            if (i != size - 1) std::cout << ", ";
        }
        if (this->size() > num_print) {
            std::cout << "...";
        }
        std::cout << "]\n";
    }

    const T& operator_(const std::vector<size_t>& indices) const {
        return operator()(indices);
    }

    Tensor(const Tensor &other) {
        *this = other;
    }

    Tensor & operator=(const Tensor &other) {
        // std::cout << "[warning] deep copy" << std::endl;
        shape_ = other.shape_;
        strides_ = other.strides_;
        data_ = other.data_;
        bitwidth = other.bitwidth;
        scale = other.scale;
        return *this; 
    }

    Tensor(Tensor &&) = default;

    Tensor &operator=(Tensor &&other) = default;

    uint64_t get_mask() const {
        return (1ULL << bitwidth) - 1;
    }

private:
    static std::string to_string_uint128(__uint128_t value) {
        if (value == 0) {
            return "0";
        }
        std::string digits;
        while (value > 0) {
            uint8_t digit = static_cast<uint8_t>(value % 10);
            digits.push_back(static_cast<char>('0' + digit));
            value /= 10;
        }
        std::reverse(digits.begin(), digits.end());
        return digits;
    }

    static std::string to_string_int128(__int128 value) {
        if (value == 0) {
            return "0";
        }
        bool negative = value < 0;
        __uint128_t magnitude = negative
            ? static_cast<__uint128_t>(-(value + 1)) + 1
            : static_cast<__uint128_t>(value);
        std::string digits = to_string_uint128(magnitude);
        if (negative) {
            return "-" + digits;
        }
        return digits;
    }

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<T> data_;

    // 计算步长
    void computeStrides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        strides_[shape_.size() - 1] = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    // 计算总大小
    size_t totalSize() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    // 计算一维索引
    size_t computeIndex(const std::vector<size_t>& indices) const {
        assert(indices.size() == shape_.size() && "Number of indices must match number of dimensions");
        size_t idx = 0;
        for (size_t i = 0; i < shape_.size(); ++i) {
            assert(indices[i] < shape_[i] && "Index out of bounds");
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

    // 辅助函数，避免重复代码
    T& operator_(const std::vector<size_t>& indices) {
        return const_cast<T&>(static_cast<const Tensor<T>&>(*this).operator()(indices));
    }
};

} // namespace Datatype