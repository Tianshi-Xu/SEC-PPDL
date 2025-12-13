import numpy as np
import struct

def float32_to_bf16_truncate(f32):
    """简单截断方法(不推荐)"""
    bits = struct.unpack('>I', struct.pack('>f', f32))[0]
    return (bits >> 16) & 0xFFFF

def float32_to_bf16_round(f32):
    """舍入到最近偶数(推荐)"""
    bits = struct.unpack('>I', struct.pack('>f', f32))[0]
    
    # 检查是否为NaN或Inf
    exp = (bits >> 23) & 0xFF
    if exp == 0xFF:
        return (bits >> 16) & 0xFFFF
    
    # 舍入到最近偶数
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bits += rounding_bias
    
    return (bits >> 16) & 0xFFFF

def bf16_to_float32(bf16):
    """BF16转回FP32"""
    bits = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', bits))[0]

def compare_methods():
    """对比两种BF16转换方法"""
    
    # 测试GeLU系数
    coe_f32 = [
        0.020848611754127593,
        -0.18352506127082727,
        0.5410550166368381,
        -0.03798164612714154,
        0.001620808531841547
    ]
    
    print("="*80)
    print("BF16转换方法对比")
    print("="*80)
    print(f"{'原始FP32':>20} | {'截断法BF16':>20} | {'舍入法BF16':>20} | {'截断误差%':>12} | {'舍入误差%':>12}")
    print("-"*80)
    
    for val in coe_f32:
        # 截断法
        bf16_trunc = float32_to_bf16_truncate(val)
        back_trunc = bf16_to_float32(bf16_trunc)
        err_trunc = abs(val - back_trunc) / (abs(val) + 1e-10) * 100
        
        # 舍入法
        bf16_round = float32_to_bf16_round(val)
        back_round = bf16_to_float32(bf16_round)
        err_round = abs(val - back_round) / (abs(val) + 1e-10) * 100
        
        print(f"{val:20.15f} | {back_trunc:20.15f} | {back_round:20.15f} | {err_trunc:12.8f} | {err_round:12.8f}")
    
    print("\n结论:")
    print("- 截断法: 直接取高16位,简单但可能有较大误差")
    print("- 舍入法: 舍入到最近偶数,精度更高,是标准做法")
    print("- 对于大多数值,两种方法结果相同或非常接近")
    print("- 建议使用舍入法以获得更好的数值稳定性")

def test_special_cases():
    """测试特殊情况"""
    print("\n" + "="*80)
    print("特殊值测试")
    print("="*80)
    
    special_values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        0.1,
        np.pi,
        np.e,
        float('inf'),
        float('-inf'),
        float('nan')
    ]
    
    print(f"{'原始值':>15} | {'BF16还原值':>15} | {'绝对误差':>15}")
    print("-"*50)
    
    for val in special_values:
        bf16 = float32_to_bf16_round(val)
        back = bf16_to_float32(bf16)
        err = abs(val - back) if not (np.isnan(val) or np.isinf(val)) else 0
        
        print(f"{str(val):>15} | {str(back):>15} | {err:15.10f}")

def analyze_precision():
    """分析BF16精度损失"""
    print("\n" + "="*80)
    print("BF16精度分析")
    print("="*80)
    
    # 在不同范围内采样
    ranges = [
        ("[-1, 1]", -1.0, 1.0, 1000),
        ("[-10, 10]", -10.0, 10.0, 1000),
        ("[0.001, 1]", 0.001, 1.0, 1000),
    ]
    
    print(f"{'范围':>15} | {'最大相对误差%':>15} | {'平均相对误差%':>15} | {'中位数误差%':>15}")
    print("-"*65)
    
    for name, start, end, num in ranges:
        values = np.linspace(start, end, num, dtype=np.float32)
        rel_errors = []
        
        for val in values:
            if abs(val) < 1e-10:
                continue
            bf16 = float32_to_bf16_round(val)
            back = bf16_to_float32(bf16)
            rel_err = abs(val - back) / abs(val) * 100
            rel_errors.append(rel_err)
        
        if rel_errors:
            max_err = np.max(rel_errors)
            avg_err = np.mean(rel_errors)
            med_err = np.median(rel_errors)
            print(f"{name:>15} | {max_err:15.10f} | {avg_err:15.10f} | {med_err:15.10f}")

def compare_with_fp32():
    """对比FP32和BF16在GeLU计算中的差异"""
    print("\n" + "="*80)
    print("FP32 vs BF16 在GeLU计算中的差异")
    print("="*80)
    
    # GeLU系数
    coe_f32 = np.array([
        0.020848611754127593,
        -0.18352506127082727,
        0.5410550166368381,
        -0.03798164612714154,
        0.001620808531841547
    ], dtype=np.float32)
    
    # 转换为BF16
    coe_bf16 = np.array([bf16_to_float32(float32_to_bf16_round(c)) for c in coe_f32], dtype=np.float32)
    
    # 测试输入
    x_values = np.linspace(-1, 3, 100, dtype=np.float32)
    
    # FP32计算
    def gelu_fp32(x):
        abs_x = np.abs(x)
        result = np.zeros_like(x)
        mask = (x >= 0) & (abs_x <= 2.7)
        result[mask] = (coe_f32[0] * abs_x[mask]**4 + 
                       coe_f32[1] * abs_x[mask]**3 + 
                       coe_f32[2] * abs_x[mask]**2 + 
                       coe_f32[3] * abs_x[mask] + 
                       coe_f32[4] + 0.5 * x[mask])
        result[x > 2.7] = x[x > 2.7]
        return result
    
    # BF16计算
    def gelu_bf16(x):
        abs_x = np.abs(x)
        result = np.zeros_like(x)
        mask = (x >= 0) & (abs_x <= 2.7)
        result[mask] = (coe_bf16[0] * abs_x[mask]**4 + 
                       coe_bf16[1] * abs_x[mask]**3 + 
                       coe_bf16[2] * abs_x[mask]**2 + 
                       coe_bf16[3] * abs_x[mask] + 
                       coe_bf16[4] + 0.5 * x[mask])
        result[x > 2.7] = x[x > 2.7]
        return result
    
    y_fp32 = gelu_fp32(x_values)
    y_bf16 = gelu_bf16(x_values)
    
    abs_err = np.abs(y_fp32 - y_bf16)
    rel_err = np.abs((y_fp32 - y_bf16) / (y_fp32 + 1e-10)) * 100
    
    print(f"最大绝对误差: {np.max(abs_err):.10f}")
    print(f"平均绝对误差: {np.mean(abs_err):.10f}")
    print(f"最大相对误差: {np.max(rel_err):.6f}%")
    print(f"平均相对误差: {np.mean(rel_err):.6f}%")
    
    # 打印一些示例点
    print("\n示例对比(前10个点):")
    print(f"{'x':>10} | {'FP32结果':>12} | {'BF16结果':>12} | {'绝对误差':>12} | {'相对误差%':>12}")
    print("-"*65)
    for i in range(min(10, len(x_values))):
        print(f"{x_values[i]:10.4f} | {y_fp32[i]:12.8f} | {y_bf16[i]:12.8f} | {abs_err[i]:12.8f} | {rel_err[i]:12.6f}")

if __name__ == "__main__":
    compare_methods()
    test_special_cases()
    analyze_precision()
    compare_with_fp32()
    
    print("\n" + "="*80)
    print("总结:")
    print("="*80)
    print("1. BF16使用舍入到最近偶数的方法精度更高")
    print("2. BF16有7位尾数精度,相当于约2-3位十进制精度")
    print("3. 对于GeLU这种多项式计算,BF16精度足够")
    print("4. C++实现应使用舍入法而非简单截断")
    print("="*80)
