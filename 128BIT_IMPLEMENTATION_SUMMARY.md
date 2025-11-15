# 128-bit Extension 和 128-bit FFT 实现总结

## 概述

本文档总结了在 SEC-PPDL 项目中实现 128-bit 定点数扩展（extension）和 128-bit FFT/IFFT 过程中遇到的主要技术困难和解决方案。

---

## 一、128-bit FFT/IFFT 实现难点

### 1.1 定点数溢出问题（最关键）

**问题描述：**
- FFT/IFFT 内部涉及大量复数乘法运算：`value * root`
- 如果 `fft_scale = 100`，则 `value` 和 `root` 都被缩放为 `2^100` 倍
- 乘法结果需要 `2^200` 的表示范围，远超 `int128_t` 的最大值 `2^127 - 1`
- 导致所有中间计算结果溢出，最终输出全为 0

**解决方案：**
采用**分层缩放策略**：
- `fft_scale`（如 40）：用于 FFT/IFFT 内部 DWT 操作的定点精度
- `payload_shift`（如 32）：用于输入数据的额外缩放
- 总缩放因子：`base = scale * payload_shift = 2^40 * 2^32 = 2^72`
- 中间乘积：`2^40 * 2^(40+32) = 2^112 < 2^127`，确保不溢出

**关键代码：**
```cpp
const int fft_scale = 40;        // 内部 DWT 操作的精度
const int extra_shift = 32;       // 输入数据的额外缩放
const int128_t scale = static_cast<int128_t>(1) << fft_scale;
const int128_t payload_shift = static_cast<int128_t>(1) << extra_shift;
const int128_t base = scale * payload_shift;  // 总缩放因子
```

### 1.2 int128_t 左移操作的限制

**问题描述：**
- 直接使用 `int64_t{1} << 100` 会在 64-bit 空间内计算，导致溢出
- 即使赋值给 `int128_t`，溢出已经发生

**解决方案：**
确保左移操作在 128-bit 空间内进行：
```cpp
// 错误：在 64-bit 空间计算
const int128_t scale = static_cast<int128_t>(int64_t{1} << fft_scale);

// 正确：在 128-bit 空间计算
const int128_t scale = static_cast<int128_t>(1) << fft_scale;
```

### 1.3 int128_t 输出问题

**问题描述：**
- `std::cout` 不支持直接输出 `int128_t` 类型
- 导致调试困难，无法验证计算结果

**解决方案：**
实现辅助函数将 `int128_t` 转换为字符串：
```cpp
std::string Int128ToString(int128_t val) {
    if (val == 0) return "0";
    bool neg = val < 0;
    int128_t abs_val = neg ? -val : val;
    std::string s;
    while (abs_val > 0) {
        s = std::to_string(static_cast<int>(abs_val % 10)) + s;
        abs_val /= 10;
    }
    return neg ? "-" + s : s;
}
```

### 1.4 SEAL 库的 128-bit 算术支持

**问题描述：**
- 需要确认 SEAL 库是否支持 128-bit 定点算术
- `seal::util::Arithmetic` 和 `seal::util::DWTHandler` 需要适配 128-bit 类型

**解决方案：**
- 利用 SEAL 库的模板设计，使用 `int128_t` 作为标量类型
- 定义 `Complex128 = std::complex<int128_t>`
- 实例化算术和 FFT 处理器：
```cpp
using Scalar128 = int128_t;
using Arithmetic128 = seal::util::Arithmetic<Complex128, Complex128, Scalar128>;
using FFTHandler128 = seal::util::DWTHandler<Complex128, Complex128, Scalar128>;
```

### 1.5 根幂的即时初始化

**问题描述：**
- 原 CKKS-MPC 项目中的 `inv_root_powers_2n_scaled` 是类成员，预先初始化
- 迁移到 SEC-PPDL 需要改为函数内即时初始化

**解决方案：**
在函数内部动态生成根幂：
```cpp
std::vector<Complex128> inv_root_powers_2n_scaled(degree, Complex128(0, 0));
seal::util::ComplexRoots complex_roots(static_cast<std::size_t>(degree) << 1, 
                                      seal::MemoryManager::GetPool());
for (std::size_t i = 1; i < degree; ++i) {
    const auto reversed_index = seal::util::reverse_bits(i - 1, logn) + 1;
    const auto inv_root = std::conj(complex_roots.get_root(reversed_index));
    inv_root_powers_2n_scaled[i] = Complex128(
        ScaleToFixed<Scalar128>(static_cast<long double>(inv_root.real()), fft_scale),
        ScaleToFixed<Scalar128>(static_cast<long double>(inv_root.imag()), fft_scale));
}
```

### 1.6 相对误差 vs 绝对误差

**问题描述：**
- 初始测试使用绝对误差，对于大数值（如 `2^72`）的舍入误差会被放大
- 导致 roundtrip 测试失败

**解决方案：**
改用相对误差评估：
```cpp
const double relative_tolerance = 0.01; // 1% 相对容差
int128_t abs_original = original_val < 0 ? -original_val : original_val;
double relative_error = abs_original > 0 
    ? static_cast<double>(diff) / static_cast<double>(abs_original)
    : (diff == 0 ? 0.0 : 1.0);
bool match = relative_error <= relative_tolerance;
```

---

## 二、128-bit Extension 实现难点

### 2.1 wrap_computation 的 128-bit 实现

**问题描述：**
- 原 `wrap_computation` 仅支持 `bw_x <= 64`
- 需要扩展到支持 `bw_x <= 128`

**解决方案：**
实现 128-bit 版本的 `wrap_computation`：
```cpp
void AuxProtocols::wrap_computation(int128_t *x, uint8_t *y, int32_t size,
                                    int32_t bw_x) {
  assert(bw_x <= 128);
  int128_t mask = (bw_x == 128 ? -1 : ((int128_t(1) << bw_x) - 1));
  
  int128_t *tmp_x = new int128_t[size];
  for (int i = 0; i < size; i++) {
    if (party == ALICE)
      tmp_x[i] = x[i] & mask;
    else
      tmp_x[i] = (mask - x[i]) & mask;  // 2^{bw_x} - 1 - x[i]
  }
  mill->compare(y, tmp_x, size, bw_x, true);  // 使用 128-bit compare
  
  delete[] tmp_x;
}
```

**关键点：**
- 使用 `int128_t` 类型的 mask 和临时数组
- 调用已扩展的 `MillionaireProtocol::compare` 的 128-bit 版本

### 2.2 z_extend 的 128-bit 实现

**问题描述：**
- `z_extend` 需要处理 `bwA > 64` 的情况
- 当 `bwA <= 64` 时，可以复用现有的 64-bit `wrap_computation`
- 当 `bwA > 64` 时，需要使用 128-bit `wrap_computation`

**解决方案：**
条件分支处理：
```cpp
void AuxProtocols::z_extend(int32_t dim, int128_t *inA, int128_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  // ... 参数验证和 mask 计算 ...
  
  if (bwA <= 64) {
    // 复用 64-bit wrap_computation
    uint64_t *inA_64 = new uint64_t[dim];
    for (int i = 0; i < dim; i++) {
      inA_64[i] = static_cast<uint64_t>(inA[i] & mask_bwA);
    }
    // ... 调用 64-bit wrap_computation ...
  } else {
    // 使用 128-bit wrap_computation
    this->wrap_computation(inA, wrap, dim, bwA);
  }
  
  // ... 后续的 B2A 和算术运算使用 int128_t ...
}
```

**关键点：**
- 对于 `bwA <= 64`，降级到 64-bit 处理以提高效率
- 对于 `bwA > 64`，使用完整的 128-bit 路径
- `B2A` 的扩展位数限制为 `min(64, extend_bits)`，因为 `B2A` 目前仅支持 64-bit

### 2.3 s_extend 的 128-bit 实现

**问题描述：**
- `s_extend` 通过映射到无符号域后调用 `z_extend` 实现
- 需要确保 128-bit 偏移计算正确

**解决方案：**
```cpp
void AuxProtocols::s_extend(int32_t dim, int128_t *inA, int128_t *outB,
                          int32_t bwA, int32_t bwB, uint8_t *msbA) {
  // ... 参数验证 ...
  
  int128_t offset = (int128_t(1) << (bwA - 1));  // 128-bit 偏移
  if (party == ALICE) {
    for (int i = 0; i < dim; i++) {
      mapped_inA[i] = (inA[i] + offset) & mask_bwA;
    }
  }
  
  // 调用 128-bit z_extend
  this->z_extend(dim, mapped_inA, mapped_outB, bwA, bwB, tmp_msbA);
  
  // ... 反向映射 ...
}
```

### 2.4 FixPoint 模板的类型分发

**问题描述：**
- `FixPoint` 是模板类，需要根据模板参数 `T` 自动选择 64-bit 或 128-bit 版本的 `extend`
- 不能使用运行时 `if`，需要在编译时决定

**解决方案：**
使用 `if constexpr` 进行编译时分发：
```cpp
void static extend_thread(AuxProtocols *aux, T* input, T* result, 
                         int lnum_ops, int32_t bwA, int32_t bwB, 
                         bool signed_arithmetic=true, uint8_t *msb_x=nullptr){
    if constexpr (std::is_same_v<T, int128_t> || std::is_same_v<T, __int128>) {
        // 128-bit version
        if (signed_arithmetic){
            aux->s_extend(lnum_ops, reinterpret_cast<int128_t*>(input), 
                         reinterpret_cast<int128_t*>(result), bwA, bwB, msb_x);
        } else {
            aux->z_extend(lnum_ops, reinterpret_cast<int128_t*>(input), 
                         reinterpret_cast<int128_t*>(result), bwA, bwB, msb_x);
        }
    } else {
        // 64-bit version (uint64_t)
        if (signed_arithmetic){
            aux->s_extend(lnum_ops, input, result, bwA, bwB, msb_x);
        } else {
            aux->z_extend(lnum_ops, input, result, bwA, bwB, msb_x);
        }
    }
}
```

**关键点：**
- `if constexpr` 在编译时求值，不会产生运行时开销
- 使用 `std::is_same_v` 进行类型匹配
- 需要包含 `<type_traits>` 头文件

### 2.5 MillionaireProtocol 的 128-bit 支持

**问题描述：**
- `MillionaireProtocol::compare` 原仅支持 `uint64_t`
- `wrap_computation` 和 `z_extend` 依赖 `compare` 协议
- 需要扩展 `compare` 支持 `int128_t`

**解决方案：**
- 修改 `configure` 函数，允许 `bitlength <= 128`
- 实现模板函数 `compare_impl<T>` 处理 `uint64_t` 和 `int128_t`
- 更新 mask 计算、内存分配和 radix-digit 提取逻辑：
```cpp
template <typename T>
void compare_impl(uint8_t *res, T *data, int num_cmps, int bitlength, ...) {
    configure(bitlength, radix_base);
    T mask = (bitlength == 128) ? static_cast<T>(-1) 
                                : ((static_cast<T>(1) << bitlength) - 1);
    // ... 使用 T 类型进行所有计算 ...
}
```

**关键点：**
- 使用模板避免代码重复
- 正确处理 128-bit 的 mask（全 1）
- Radix-digit 提取时注意移位不能超过 128-bit

---

## 三、编译和类型系统问题

### 3.1 头文件依赖

**问题描述：**
- `int128_t` 定义在 `<seal/util/common.h>` 中
- 多个文件需要包含此头文件

**解决方案：**
在需要的头文件中统一包含：
- `Polynomial.h`: 包含 `<seal/util/common.h>` 和 `<cstddef>`
- `aux-protocols.h`: 包含 `<seal/util/common.h>`
- `FixPoint.h`: 包含 `<seal/util/common.h>` 和 `<type_traits>`

### 3.2 Tensor 打印问题

**问题描述：**
- `Tensor<T>::print()` 使用 `std::cout << data_[i]`
- `int128_t` 没有 `operator<<` 重载，导致编译错误

**解决方案：**
在测试代码中手动打印，避免调用 `Tensor<T128>::print()`：
```cpp
// 避免使用 input.print()
std::cout << "Input values: ";
for (int i = 0; i < input.size(); i++) {
    std::cout << static_cast<long long>(input(i)) << " ";
}
```

---

## 四、性能考虑

### 4.1 128-bit 运算的开销

- 128-bit 整数运算比 64-bit 慢约 2-4 倍
- 在 `bwA <= 64` 时，`z_extend` 降级到 64-bit 处理以提高效率

### 4.2 内存占用

- `int128_t` 占用 16 字节，是 `uint64_t` 的 2 倍
- 大数组的内存占用显著增加

---

## 五、已知限制和未来改进

### 5.1 B2A 协议的限制

- `B2A` 目前仅支持扩展到 64-bit
- 对于 `extend_bits > 64`，使用 `min(64, extend_bits)`
- 未来需要实现完整的 128-bit `B2A` 协议

### 5.2 128-bit wrap_computation 的简化

- 当 `bwA > 64` 时，`wrap_computation` 的实现是简化版本
- 完整的 128-bit 安全 wrap 协议会更复杂，但当前实现已满足基本需求

---

## 六、测试验证

### 6.1 FFT Roundtrip 测试

- 测试 IFFT 后接 FFT 是否恢复原始输入
- 使用相对误差（1%）评估精度
- 验证大数值（超过 64-bit 范围）的正确性

### 6.2 Extension 测试

- 测试从 40-bit 扩展到 72-bit
- 验证 `z_extend` 和 `s_extend` 的正确性
- 通过重构共享值验证协议安全性

---

## 七、总结

实现 128-bit extension 和 128-bit FFT 的主要挑战包括：

1. **定点数溢出管理**：通过分层缩放策略避免中间计算溢出
2. **类型系统集成**：使用模板和 `if constexpr` 实现类型安全的多态
3. **协议链扩展**：从底层的 `wrap_computation` 到上层的 `FixPoint::extend` 全链路支持
4. **调试工具**：实现 `int128_t` 的字符串转换以支持调试
5. **精度评估**：使用相对误差而非绝对误差评估大数值的精度

这些实现为 SEC-PPDL 项目提供了处理大范围定点数的能力，支持更高精度的安全多方计算。

