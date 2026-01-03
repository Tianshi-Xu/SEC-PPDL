import numpy as np
import matplotlib.pyplot as plt


class FixedPoint:
    """
    定点数类,用于模拟定点数运算
    """
    def __init__(self, integer_bits, fractional_bits):
        """
        Args:
            integer_bits: 整数部分位数
            fractional_bits: 小数部分位数
        """
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.scale = 2 ** fractional_bits
        self.max_val = (2 ** (self.total_bits - 1) - 1) / self.scale
        self.min_val = -(2 ** (self.total_bits - 1)) / self.scale
    
    def quantize(self, x):
        """将浮点数量化为定点数"""
        # 转换为定点表示
        fixed = np.round(x * self.scale).astype(np.int64)
        # 截断到范围内
        max_int = 2 ** (self.total_bits - 1) - 1
        min_int = -(2 ** (self.total_bits - 1))
        fixed = np.clip(fixed, min_int, max_int)
        return fixed
    
    def dequantize(self, fixed):
        """将定点数还原为浮点数"""
        return fixed.astype(np.float64) / self.scale
    
    def fixed_add(self, a_fixed, b_fixed):
        """定点数加法"""
        return a_fixed + b_fixed
    
    def fixed_mul(self, a_fixed, b_fixed):
        """定点数乘法,需要处理精度损失"""
        result = (a_fixed.astype(np.int64) * b_fixed.astype(np.int64))
        # 右移fractional_bits位以保持定点格式
        result = result >> self.fractional_bits
        return result.astype(np.int64)
    
    def fixed_div(self, a_fixed, b_fixed):
        """定点数除法"""
        # 左移fractional_bits位以保持精度
        result = (a_fixed.astype(np.int64) << self.fractional_bits) // b_fixed.astype(np.int64)
        return result.astype(np.int64)
    
    def fixed_power(self, base_fixed, n):
        """定点数幂运算,使用快速幂算法"""
        if n == 0:
            return self.quantize(np.ones_like(self.dequantize(base_fixed)))
        
        result = self.quantize(np.ones_like(self.dequantize(base_fixed)))
        power = base_fixed.copy()
        
        while n > 0:
            if n & 1:
                result = self.fixed_mul(result, power)
            power = self.fixed_mul(power, power)
            n >>= 1
        
        return result


def exp_normal(x):
    """
    正常的exp函数实现
    
    Args:
        x: 输入值,支持标量或numpy数组
    
    Returns:
        exp(x)的结果
    """
    return np.exp(x)


def exp_polynomial_approx(x):
    """
    多项式近似的exp函数实现
    使用公式: exp(x) = (1 + x/(2^6))^(2^6)
    
    这个近似方法基于(1+x/n)^n在n趋向无穷时等于e^x的极限
    这里选择n=2^6=64作为平衡精度和计算效率的折中
    
    Args:
        x: 输入值,范围应该在[-13, 0]之间,数值类型为fp32
    
    Returns:
        exp(x)的近似结果
    """
    # 确保输入为fp32类型
    x = np.float32(x)
    
    # n = 2^6 = 64
    n = 64
    
    # 计算 (1 + x/n)^n
    result = np.power(np.float32(1.0) + x / np.float32(n), n)
    
    return result


def exp_fixed_point(x, integer_bits=8, fractional_bits=24, *, fp=None, x_is_fixed=False, return_fixed=False):
    """
    使用定点数实现的exp函数近似, 支持直接传入定点数数据
    使用公式: exp(x) = (1 + x/(2^6))^(2^6)
    
    Args:
        x: 输入值,范围应该在[-13, 0]之间
        integer_bits: 定点数整数部分位数
        fractional_bits: 定点数小数部分位数
        fp: 复用的FixedPoint实例
        x_is_fixed: 输入是否已经是定点数格式
        return_fixed: 是否直接返回定点数结果
    
    Returns:
        exp(x)的定点数近似结果(浮点或定点数)
    """
    # 创建或复用定点数对象
    fp = fp or FixedPoint(integer_bits, fractional_bits)
    
    # 将输入转换为定点数
    if x_is_fixed:
        x_fixed = np.asarray(x, dtype=np.int64)
    else:
        x_fixed = fp.quantize(np.asarray(x, dtype=np.float64))
    
    # 处理标量输入
    squeeze = False
    if x_fixed.ndim == 0:
        x_fixed = np.expand_dims(x_fixed, 0)
        squeeze = True
    
    # n = 2^6 = 64
    n = 64
    shape = x_fixed.shape
    n_fixed = fp.quantize(np.full(shape, n, dtype=np.float64))
    
    # 计算 x/n
    x_div_n = fp.fixed_div(x_fixed, n_fixed)
    
    # 计算 1 + x/n
    one_fixed = fp.quantize(np.ones(shape, dtype=np.float64))
    base = fp.fixed_add(one_fixed, x_div_n)
    
    # 计算 (1 + x/n)^64
    result_fixed = fp.fixed_power(base, n)
    
    if squeeze:
        result_fixed = np.squeeze(result_fixed, axis=0)
    
    if return_fixed:
        return result_fixed
    
    # 转换回浮点数
    result = fp.dequantize(result_fixed)
    
    return result.astype(np.float32)


def test_exp_functions(integer_bits=8, fractional_bits=24):
    """
    测试和比较三个exp函数的精度
    
    Args:
        integer_bits: 定点数整数部分位数
        fractional_bits: 定点数小数部分位数
    """
    # 生成测试数据: 在[-13, 0]范围内均匀采样
    x_values = np.linspace(-13, 0, 1000, dtype=np.float32)
    
    # 计算三个函数的结果
    y_normal = exp_normal(x_values)
    y_approx = exp_polynomial_approx(x_values)
    y_fixed = exp_fixed_point(x_values, integer_bits, fractional_bits)
    
    # 计算绝对误差和相对误差
    abs_error_poly = np.abs(y_normal - y_approx)
    rel_error_poly = np.abs((y_normal - y_approx) / (y_normal + 1e-10))
    
    abs_error_fixed = np.abs(y_normal - y_fixed)
    rel_error_fixed = np.abs((y_normal - y_fixed) / (y_normal + 1e-10))
    
    # 打印统计信息
    print("=" * 70)
    print("Exp函数精度测试报告 (三种方法对比)")
    print("=" * 70)
    print(f"测试范围: x ∈ [-13, 0]")
    print(f"测试点数: {len(x_values)}")
    print(f"定点数配置: Q{integer_bits}.{fractional_bits} (总{integer_bits+fractional_bits}位)")
    print()
    
    print("【多项式近似 FP32】绝对误差统计:")
    print(f"  最大绝对误差: {np.max(abs_error_poly):.10f}")
    print(f"  平均绝对误差: {np.mean(abs_error_poly):.10f}")
    print()
    
    print("【多项式近似 FP32】相对误差统计:")
    print(f"  最大相对误差: {np.max(rel_error_poly):.10f} ({np.max(rel_error_poly)*100:.6f}%)")
    print(f"  平均相对误差: {np.mean(rel_error_poly):.10f} ({np.mean(rel_error_poly)*100:.6f}%)")
    print()
    
    print(f"【定点数 Q{integer_bits}.{fractional_bits}】绝对误差统计:")
    print(f"  最大绝对误差: {np.max(abs_error_fixed):.10f}")
    print(f"  平均绝对误差: {np.mean(abs_error_fixed):.10f}")
    print()
    
    print(f"【定点数 Q{integer_bits}.{fractional_bits}】相对误差统计:")
    print(f"  最大相对误差: {np.max(rel_error_fixed):.10f} ({np.max(rel_error_fixed)*100:.6f}%)")
    print(f"  平均相对误差: {np.mean(rel_error_fixed):.10f} ({np.mean(rel_error_fixed)*100:.6f}%)")
    print("=" * 70)
    
    return x_values, y_normal, y_approx, y_fixed, abs_error_poly, rel_error_poly, abs_error_fixed, rel_error_fixed


def test_different_bitwidths():
    """
    测试不同位宽的定点数配置
    """
    print("\n" + "=" * 115)
    print("不同定点数位宽配置对比")
    print("=" * 115)
    
    # 生成测试数据
    x_values = np.linspace(-13, 0, 1000, dtype=np.float32)
    y_normal = exp_normal(x_values)
    
    # 测试不同的配置: (integer_bits, fractional_bits)
    # 重点测试24-48位范围内的各种组合,特别关注整数位需求
    configs = [
        # 16位配置 - 测试极少整数位
        (1, 15),   # 16位 - 只有1位整数
        (2, 14),   # 16位 - 2位整数
        (3, 13),   # 16位 - 3位整数
        (4, 12),   # 16位 - 4位整数
        
        # 24位配置
        (1, 23),   # 24位 - 只有1位整数
        (2, 22),   # 24位 - 2位整数
        (3, 21),   # 24位 - 3位整数
        (4, 20),   # 24位
        (6, 18),   # 24位
        (8, 16),   # 24位
        (10, 14),  # 24位
        (12, 12),  # 24位
        
        # 28位配置
        (2, 26),   # 28位 - 2位整数
        (4, 24),   # 28位 - 4位整数
        (6, 22),   # 28位
        (8, 20),   # 28位
        (10, 18),  # 28位
        (12, 16),  # 28位
        
        # 32位配置
        (1, 31),   # 32位 - 只有1位整数
        (2, 30),   # 32位 - 2位整数
        (3, 29),   # 32位 - 3位整数
        (4, 28),   # 32位 - 4位整数
        (6, 26),   # 32位
        (8, 24),   # 32位
        (10, 22),  # 32位
        (12, 20),  # 32位
        (14, 18),  # 32位
        (16, 16),  # 32位
        
        # 36位配置
        (2, 34),   # 36位 - 2位整数
        (4, 32),   # 36位 - 4位整数
        (8, 28),   # 36位
        (10, 26),  # 36位
        (12, 24),  # 36位
        (16, 20),  # 36位
        
        # 40位配置
        (2, 38),   # 40位 - 2位整数
        (4, 36),   # 40位 - 4位整数
        (8, 32),   # 40位
        (10, 30),  # 40位
        (12, 28),  # 40位
        (16, 24),  # 40位
        (20, 20),  # 40位
        
        # 48位配置
        (2, 46),   # 48位 - 2位整数
        (4, 44),   # 48位 - 4位整数
        (8, 40),   # 48位
        (12, 36),  # 48位
        (16, 32),  # 48位
        (20, 28),  # 48位
        (24, 24),  # 48位
    ]
    
    results = []
    
    print(f"\n{'总位数':>8} | {'整数位':>8} | {'小数位':>8} | {'最大绝对误差':>15} | {'平均绝对误差':>15} | {'最大相对误差%':>15} | {'平均相对误差%':>15}")
    print("-" * 115)
    
    for int_bits, frac_bits in configs:
        try:
            y_fixed = exp_fixed_point(x_values, int_bits, frac_bits)
            
            abs_error = np.abs(y_normal - y_fixed)
            rel_error = np.abs((y_normal - y_fixed) / (y_normal + 1e-10))
            
            total_bits = int_bits + frac_bits
            max_abs = np.max(abs_error)
            mean_abs = np.mean(abs_error)
            max_rel = np.max(rel_error) * 100
            mean_rel = np.mean(rel_error) * 100
            
            print(f"{total_bits:8d} | {int_bits:8d} | {frac_bits:8d} | {max_abs:15.10f} | {mean_abs:15.10f} | {max_rel:15.6f} | {mean_rel:15.6f}")
            
            results.append({
                'config': (int_bits, frac_bits),
                'total_bits': total_bits,
                'max_abs_error': max_abs,
                'mean_abs_error': mean_abs,
                'max_rel_error': max_rel,
                'mean_rel_error': mean_rel
            })
        except Exception as e:
            print(f"{int_bits+frac_bits:8d} | {int_bits:8d} | {frac_bits:8d} | Error: {str(e)[:50]}")
    
    print("=" * 115)
    
    # 找出最优配置
    if results:
        best_by_mean_abs = min(results, key=lambda x: x['mean_abs_error'])
        best_by_mean_rel = min(results, key=lambda x: x['mean_rel_error'])
        
        print("\n【全局最优配置】")
        print(f"✓ 最佳平均绝对误差: Q{best_by_mean_abs['config'][0]}.{best_by_mean_abs['config'][1]} "
              f"(总{best_by_mean_abs['total_bits']}位, 平均误差={best_by_mean_abs['mean_abs_error']:.10f})")
        print(f"✓ 最佳平均相对误差: Q{best_by_mean_rel['config'][0]}.{best_by_mean_rel['config'][1]} "
              f"(总{best_by_mean_rel['total_bits']}位, 平均误差={best_by_mean_rel['mean_rel_error']:.6f}%)")
        
        # 按位宽分组找最优配置
        print("\n【不同位宽范围的最优配置】")
        
        # 24位配置
        range_24 = [r for r in results if r['total_bits'] == 24]
        if range_24:
            best_24 = min(range_24, key=lambda x: x['mean_abs_error'])
            print(f"✓ 24位最优: Q{best_24['config'][0]}.{best_24['config'][1]} "
                  f"(平均绝对误差={best_24['mean_abs_error']:.10f}, 平均相对误差={best_24['mean_rel_error']:.6f}%)")
        
        # 28位配置
        range_28 = [r for r in results if r['total_bits'] == 28]
        if range_28:
            best_28 = min(range_28, key=lambda x: x['mean_abs_error'])
            print(f"✓ 28位最优: Q{best_28['config'][0]}.{best_28['config'][1]} "
                  f"(平均绝对误差={best_28['mean_abs_error']:.10f}, 平均相对误差={best_28['mean_rel_error']:.6f}%)")
        
        # 32位配置
        range_32 = [r for r in results if r['total_bits'] == 32]
        if range_32:
            best_32 = min(range_32, key=lambda x: x['mean_abs_error'])
            print(f"✓ 32位最优: Q{best_32['config'][0]}.{best_32['config'][1]} "
                  f"(平均绝对误差={best_32['mean_abs_error']:.10f}, 平均相对误差={best_32['mean_rel_error']:.6f}%)")
        
        # 36位配置
        range_36 = [r for r in results if r['total_bits'] == 36]
        if range_36:
            best_36 = min(range_36, key=lambda x: x['mean_abs_error'])
            print(f"✓ 36位最优: Q{best_36['config'][0]}.{best_36['config'][1]} "
                  f"(平均绝对误差={best_36['mean_abs_error']:.10f}, 平均相对误差={best_36['mean_rel_error']:.6f}%)")
        
        # 40位配置
        range_40 = [r for r in results if r['total_bits'] == 40]
        if range_40:
            best_40 = min(range_40, key=lambda x: x['mean_abs_error'])
            print(f"✓ 40位最优: Q{best_40['config'][0]}.{best_40['config'][1]} "
                  f"(平均绝对误差={best_40['mean_abs_error']:.10f}, 平均相对误差={best_40['mean_rel_error']:.6f}%)")
        
        # 48位配置
        range_48 = [r for r in results if r['total_bits'] == 48]
        if range_48:
            best_48 = min(range_48, key=lambda x: x['mean_abs_error'])
            print(f"✓ 48位最优: Q{best_48['config'][0]}.{best_48['config'][1]} "
                  f"(平均绝对误差={best_48['mean_abs_error']:.10f}, 平均相对误差={best_48['mean_rel_error']:.6f}%)")
        
        # 推荐性价比最高的配置
        print("\n【推荐配置(性价比考虑)】")
        target_range = [r for r in results if 24 <= r['total_bits'] <= 32]
        if target_range:
            best_target = min(target_range, key=lambda x: x['mean_abs_error'])
            print(f"✓ 24-32位范围推荐: Q{best_target['config'][0]}.{best_target['config'][1]} "
                  f"(总{best_target['total_bits']}位)")
            print(f"  - 平均绝对误差: {best_target['mean_abs_error']:.10f}")
            print(f"  - 平均相对误差: {best_target['mean_rel_error']:.6f}%")
            print(f"  - 最大绝对误差: {best_target['max_abs_error']:.10f}")
            print(f"  - 说明: 平衡精度和硬件成本,适合实际LLM应用")
        
        # 分析整数位需求
        print("\n【整数位需求分析】")
        print("比较相同总位数下,不同整数位配置的精度:")
        
        # 分析32位配置中不同整数位的影响
        bits_32 = [r for r in results if r['total_bits'] == 32]
        if bits_32:
            bits_32_sorted = sorted(bits_32, key=lambda x: x['config'][0])
            print("\n32位定点数 - 不同整数位配置对比:")
            print(f"{'整数位':>8} | {'小数位':>8} | {'平均绝对误差':>15} | {'平均相对误差%':>15}")
            print("-" * 60)
            for r in bits_32_sorted:
                print(f"{r['config'][0]:8d} | {r['config'][1]:8d} | {r['mean_abs_error']:15.10f} | {r['mean_rel_error']:15.6f}")
            
            # 找出最少整数位的最优配置
            min_int_32 = min(bits_32_sorted, key=lambda x: x['mean_abs_error'])
            print(f"\n✓ 结论: 32位下最优配置是 Q{min_int_32['config'][0]}.{min_int_32['config'][1]}")
            if min_int_32['config'][0] <= 4:
                print(f"  说明: 只需{min_int_32['config'][0]}位整数位即可达到最优精度!")
                print(f"       因为exp(x)输出在[0,1]范围,不需要8位整数位")
    
    return results


def plot_comparison(x_values, y_normal, y_approx, y_fixed, abs_error_poly, rel_error_poly, abs_error_fixed, rel_error_fixed, 
                    integer_bits=8, fractional_bits=24):
    """
    绘制对比图表(三种方法)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 子图1: exp函数值对比
    axes[0, 0].plot(x_values, y_normal, 'b-', label='Normal exp(x)', linewidth=2)
    axes[0, 0].plot(x_values, y_approx, 'r--', label='Polynomial approx (FP32)', linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(x_values, y_fixed, 'g:', label=f'Fixed-point Q{integer_bits}.{fractional_bits}', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('exp(x)')
    axes[0, 0].set_title('Exp Function Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 多项式近似的绝对误差
    axes[0, 1].plot(x_values, abs_error_poly, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Polynomial Approx (FP32): Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 定点数的绝对误差
    axes[0, 2].plot(x_values, abs_error_fixed, 'g-', linewidth=2)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('Absolute Error')
    axes[0, 2].set_title(f'Fixed-point Q{integer_bits}.{fractional_bits}: Absolute Error')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 子图4: 对数尺度下的exp函数值对比
    axes[1, 0].semilogy(x_values, y_normal, 'b-', label='Normal exp(x)', linewidth=2)
    axes[1, 0].semilogy(x_values, y_approx, 'r--', label='Polynomial approx (FP32)', linewidth=1.5, alpha=0.8)
    axes[1, 0].semilogy(x_values, y_fixed, 'g:', label=f'Fixed-point Q{integer_bits}.{fractional_bits}', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('exp(x) (log scale)')
    axes[1, 0].set_title('Exp Function Comparison (Log Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图5: 多项式近似的相对误差
    axes[1, 1].plot(x_values, rel_error_poly * 100, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Relative Error (%)')
    axes[1, 1].set_title('Polynomial Approx (FP32): Relative Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 子图6: 定点数的相对误差
    axes[1, 2].plot(x_values, rel_error_fixed * 100, 'g-', linewidth=2)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Relative Error (%)')
    axes[1, 2].set_title(f'Fixed-point Q{integer_bits}.{fractional_bits}: Relative Error')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exp_comparison_3methods.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 'exp_comparison_3methods.png'")
    plt.show()


def test_specific_values(integer_bits=8, fractional_bits=24):
    """
    测试一些特定值(三种方法)
    """
    print("\n特定值测试 (三种方法对比):")
    print("-" * 100)
    test_values = [-13, -10, -7.5, -5, -2.5, -1, -0.5, -0.1, 0]
    
    print(f"{'x':>8} | {'exp(x)':>15} | {'poly_approx':>15} | {'fixed_approx':>15} | {'poly_err%':>12} | {'fixed_err%':>12}")
    print("-" * 100)
    
    for x in test_values:
        x_fp32 = np.float32(x)
        normal = exp_normal(x_fp32)
        approx = exp_polynomial_approx(x_fp32)
        fixed = exp_fixed_point(x_fp32, integer_bits, fractional_bits)
        
        poly_rel_err = np.abs((normal - approx) / (normal + 1e-10)) * 100
        fixed_rel_err = np.abs((normal - fixed) / (normal + 1e-10)) * 100
        
        print(f"{x:8.2f} | {normal:15.10f} | {approx:15.10f} | {fixed:15.10f} | {poly_rel_err:12.6f} | {fixed_rel_err:12.6f}")


def softmax_fixed_point(logits, integer_bits=8, fractional_bits=24, axis=-1, return_fixed=False):
    """使用完全定点路径计算softmax"""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 0:
        raise ValueError("Softmax输入至少需要一维")
    fp = FixedPoint(integer_bits, fractional_bits)
    moved = np.moveaxis(logits, axis, -1)
    original_shape = moved.shape
    flattened = moved.reshape(-1, original_shape[-1])
    probs_fixed = np.zeros_like(flattened, dtype=np.int64)
    for idx, row in enumerate(flattened):
        row_fixed = fp.quantize(row)
        max_fixed = np.max(row_fixed)
        shifted_fixed = row_fixed - max_fixed
        exp_vals_fixed = exp_fixed_point(shifted_fixed, fp=fp, x_is_fixed=True, return_fixed=True)
        sum_fixed = np.sum(exp_vals_fixed, dtype=np.int64)
        if sum_fixed == 0:
            raise ZeroDivisionError("Softmax分母为0, 请检查输入范围")
        denom_fixed = np.full_like(exp_vals_fixed, sum_fixed)
        probs_fixed[idx] = fp.fixed_div(exp_vals_fixed, denom_fixed)
    probs_fixed = probs_fixed.reshape(original_shape)
    probs_fixed = np.moveaxis(probs_fixed, -1, axis)
    if return_fixed:
        return probs_fixed
    return fp.dequantize(probs_fixed).astype(np.float32)


def plot_softmax_comparison(logits, softmax_normal, softmax_fixed, integer_bits=8, fractional_bits=24,
                            filename='softmax_fixed_vs_float.png'):
    """Draw a smoothed single-chart comparison between float and fixed-point softmax."""
    logits = np.asarray(logits).reshape(-1)
    softmax_normal = np.asarray(softmax_normal).reshape(-1)
    softmax_fixed = np.asarray(softmax_fixed).reshape(-1)
    if logits.size != softmax_normal.size:
        logits = np.linspace(0, softmax_normal.size - 1, softmax_normal.size)
    sort_idx = np.argsort(logits)
    logits_sorted = logits[sort_idx]
    normal_sorted = softmax_normal[sort_idx]
    fixed_sorted = softmax_fixed[sort_idx]
    x_dense = np.linspace(logits_sorted[0], logits_sorted[-1], 300)
    normal_dense = np.interp(x_dense, logits_sorted, normal_sorted)
    fixed_dense = np.interp(x_dense, logits_sorted, fixed_sorted)
    error_dense = np.abs(normal_dense - fixed_dense)
    fig, ax_prob = plt.subplots(figsize=(10, 6))
    prob_line = ax_prob.plot(x_dense, normal_dense, label='Softmax FP32', linewidth=2)
    fixed_line = ax_prob.plot(x_dense, fixed_dense, linestyle='--', label=f'Softmax Q{integer_bits}.{fractional_bits}', linewidth=2)
    ax_prob.set_xlabel('Input logits (x)')
    ax_prob.set_ylabel('Probability (y)')
    ax_prob.set_title('Softmax Comparison (Float vs Fixed-point)')
    ax_prob.grid(True, alpha=0.3)
    ax_err = ax_prob.twinx()
    err_fill = ax_err.fill_between(x_dense, 0, error_dense, color='tab:red', alpha=0.25, label='Absolute error |Δy|')
    ax_err.set_ylabel('Absolute error |Δy|')
    lines = prob_line + fixed_line + [err_fill]
    labels = ['Softmax FP32', f'Softmax Q{integer_bits}.{fractional_bits}', 'Absolute error |Δy|']
    ax_prob.legend(lines, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSoftmax比较图已保存为 '{filename}'")
    plt.show()


def test_softmax_scenario(integer_bits=8, fractional_bits=24, plot=True):
    """
    模拟LLM中softmax的场景测试(三种方法)
    """
    print("\n\nSoftmax场景测试 (模拟LLM - 三种方法对比):")
    print("=" * 95)
    
    # 模拟一个典型的logits向量(经过减去最大值的稳定化处理)
    logits = np.array([-0.5, -1.2, -3.5, -0.1, -2.8, -5.0, -0.3, -7.2], dtype=np.float32)
    
    print(f"输入 logits: {logits}")
    print(f"定点数配置: Q{integer_bits}.{fractional_bits}")
    print()
    
    # 使用正常exp计算softmax
    exp_logits_normal = exp_normal(logits)
    softmax_normal = exp_logits_normal / np.sum(exp_logits_normal)
    
    # 使用多项式近似exp计算softmax
    exp_logits_approx = exp_polynomial_approx(logits)
    softmax_approx = exp_logits_approx / np.sum(exp_logits_approx)
    
    # 使用全定点softmax
    softmax_fixed = softmax_fixed_point(logits, integer_bits, fractional_bits)
    
    print("Softmax结果对比:")
    print(f"{'Idx':>4} | {'Normal':>12} | {'Poly_FP32':>12} | {'Fixed':>12} | {'Poly_Err%':>11} | {'Fixed_Err%':>12}")
    print("-" * 95)
    
    for i in range(len(logits)):
        poly_diff = np.abs((softmax_normal[i] - softmax_approx[i]) / (softmax_normal[i] + 1e-10)) * 100
        fixed_diff = np.abs((softmax_normal[i] - softmax_fixed[i]) / (softmax_normal[i] + 1e-10)) * 100
        print(f"{i:4d} | {softmax_normal[i]:12.10f} | {softmax_approx[i]:12.10f} | {softmax_fixed[i]:12.10f} | "
              f"{poly_diff:11.6f} | {fixed_diff:12.6f}")
    
    print()
    print(f"【多项式近似 FP32】Softmax最大绝对误差: {np.max(np.abs(softmax_normal - softmax_approx)):.10f}")
    print(f"【多项式近似 FP32】Softmax平均绝对误差: {np.mean(np.abs(softmax_normal - softmax_approx)):.10f}")
    print(f"【定点数 Q{integer_bits}.{fractional_bits}】Softmax最大绝对误差: {np.max(np.abs(softmax_normal - softmax_fixed)):.10f}")
    print(f"【定点数 Q{integer_bits}.{fractional_bits}】Softmax平均绝对误差: {np.mean(np.abs(softmax_normal - softmax_fixed)):.10f}")

    if plot:
        plot_softmax_comparison(logits, softmax_normal, softmax_fixed, integer_bits, fractional_bits)

    return softmax_normal, softmax_fixed


if __name__ == "__main__":
    # 第一阶段: 使用默认配置Q8.24进行初步测试
    default_int_bits = 8
    default_frac_bits = 24
    
    print("\n" + "="*70)
    print(f"第一阶段: 使用默认配置 Q{default_int_bits}.{default_frac_bits} 进行初步测试")
    print("="*70)
    
    # 运行主测试
    x_values, y_normal, y_approx, y_fixed, abs_error_poly, rel_error_poly, abs_error_fixed, rel_error_fixed = \
        test_exp_functions(default_int_bits, default_frac_bits)
    
    # 测试特定值
    test_specific_values(default_int_bits, default_frac_bits)
    
    # 测试softmax场景
    test_softmax_scenario(default_int_bits, default_frac_bits)
    
    # 第二阶段: 测试不同位宽配置,找出最优配置
    print("\n" + "="*70)
    print("第二阶段: 测试24-48位范围内的不同配置")
    print("="*70)
    bitwidth_results = test_different_bitwidths()
    
    # 第三阶段: 使用推荐的最优配置Q8.24重新测试和绘图
    recommended_int_bits = 8
    recommended_frac_bits = 24
    
    print("\n" + "="*70)
    print(f"第三阶段: 使用推荐配置 Q{recommended_int_bits}.{recommended_frac_bits} 进行详细测试和可视化")
    print("="*70)
    
    # 使用推荐配置重新计算
    x_values_rec, y_normal_rec, y_approx_rec, y_fixed_rec, abs_error_poly_rec, rel_error_poly_rec, abs_error_fixed_rec, rel_error_fixed_rec = \
        test_exp_functions(recommended_int_bits, recommended_frac_bits)
    
    # 测试特定值
    test_specific_values(recommended_int_bits, recommended_frac_bits)
    
    # 测试softmax场景
    test_softmax_scenario(recommended_int_bits, recommended_frac_bits)
    
    # 绘制对比图表(使用推荐配置)
    plot_comparison(x_values_rec, y_normal_rec, y_approx_rec, y_fixed_rec, 
                   abs_error_poly_rec, rel_error_poly_rec, abs_error_fixed_rec, rel_error_fixed_rec,
                   recommended_int_bits, recommended_frac_bits)
