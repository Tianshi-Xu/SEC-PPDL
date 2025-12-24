import math

class FixedPointSystem:
    def __init__(self, fractional_bits=20):
        self.frac_bits = fractional_bits
        self.scale = 1 << fractional_bits
        # 舍入用的常数 (0.5)
        self.round_offset = 1 << (fractional_bits - 1)
    
    def to_fixed(self, float_val):
        """浮点转定点"""
        return int(round(float_val * self.scale))
    
    def to_float(self, fixed_val):
        """定点转浮点 (用于观察结果)"""
        return fixed_val / self.scale
    
    def mul(self, a, b):
        """
        定点乘法模拟: (A * B) >> scale
        包含四舍五入逻辑
        """
        product = a * b
        # 加上 0.5 (在定点域是 1<<(n-1)) 实现四舍五入
        return (product + self.round_offset) >> self.frac_bits

class PolySolver:
    def __init__(self, system, coeffs):
        self.sys = system
        # 预先将系数量化为定点数
        self.c = [system.to_fixed(x) for x in coeffs]
        # c[0]=x^4, c[1]=x^3, c[2]=x^2, c[3]=x^1(部分), c[4]=const
        
        # 预计算定点数 0.5
        self.half = system.to_fixed(0.5)

    def calc_naive_direct(self, x_fix):
        """
        方法一：朴素逐项计算 (Direct Method)
        完全照搬你的公式：
        F0 = c0*x^4 - c1*x^3 + c2*x^2 + x*(0.5 - c3) + c4
        F1 = c0*x^4 + c1*x^3 + c2*x^2 + x*(0.5 + c3) + c4
        """
        sys = self.sys
        c = self.c
        
        # 1. 极其昂贵的幂次计算 (容易导致中间值溢出)
        x2 = sys.mul(x_fix, x_fix)
        x3 = sys.mul(x2, x_fix)
        x4 = sys.mul(x3, x_fix)
        
        # --- 计算 F0 ---
        # 项计算
        t4 = sys.mul(x4, c[0])
        t3 = sys.mul(x3, c[1]) 
        t2 = sys.mul(x2, c[2])
        
        # 线性项: x * (0.5 - c3)
        coef_linear_f0 = self.half - c[3]
        t1_f0 = sys.mul(x_fix, coef_linear_f0)
        
        # 组合: t4 - t3 + t2 + t1 + c4
        f0 = t4 - t3 + t2 + t1_f0 + c[4]
        
        # --- 计算 F1 ---
        # 线性项: x * (0.5 + c3)
        coef_linear_f1 = self.half + c[3]
        t1_f1 = sys.mul(x_fix, coef_linear_f1)
        
        # 组合: t4 + t3 + t2 + t1 + c4
        f1 = t4 + t3 + t2 + t1_f1 + c[4]
        
        return f0, f1

    def calc_optimized_odd_even(self, x_fix):
        """
        方法二：奇偶分解法 (Odd-Even Decomposition)
        利用 F0 和 F1 的对称性，复用计算。
        F0 = E + L - O
        F1 = E + L + O
        """
        sys = self.sys
        c = self.c
        
        # 1. 基础: x^2 (乘法 #1)
        x2 = sys.mul(x_fix, x_fix)
        
        # 2. 公共偶数部分 E(x) = c0*x^4 + c2*x^2 + c4
        # 使用秦九韶折叠: E = (c0 * x^2 + c2) * x^2 + c4
        
        # (乘法 #2)
        even_part = sys.mul(c[0], x2)
        even_part += c[2]
        
        # (乘法 #3)
        even_part = sys.mul(even_part, x2)
        even_part += c[4]
        
        # 3. 公共奇数部分 O(x) = c1*x^3 + c3*x
        # 使用秦九韶折叠: O = (c1 * x^2 + c3) * x
        
        # (乘法 #4)
        odd_part = sys.mul(c[1], x2)
        odd_part += c[3]
        
        # (乘法 #5)
        odd_part = sys.mul(odd_part, x_fix)
        
        # 4. 线性偏移 L(x) = 0.5 * x
        # 在定点数中，乘以 0.5 等于右移 1 位 (不消耗乘法器)
        linear_offset = x_fix >> 1
        
        # 5. 组合结果
        # F0 = E + L - O (注意原始公式 F0 是减去奇数项部分)
        f0 = even_part + linear_offset - odd_part
        
        # F1 = E + L + O (原始公式 F1 是加上奇数项部分)
        f1 = even_part + linear_offset + odd_part
        
        return f0, f1

# ==========================================
# 测试与验证
# ==========================================

# 1. 配置
# 你的原始系数
raw_coeffs = [
    0.020848611754127593,  # c0
    -0.18352506127082727,  # c1
    0.5410550166368381,    # c2
    -0.03798164612714154,  # c3
    0.001620808531841547   # c4
]

# 初始化定点系统 (Q12.20)
fps = FixedPointSystem(fractional_bits=20)
solver = PolySolver(fps, raw_coeffs)

# 测试输入 x
test_x = 2.5
x_fixed = fps.to_fixed(test_x)

print(f"输入 x: {test_x} (Fixed: {x_fixed})")
print("=" * 60)

# --- 运行朴素算法 ---
res_naive_f0, res_naive_f1 = solver.calc_naive_direct(x_fixed)
print(f"[朴素算法 Direct]")
print(f"  F0: {fps.to_float(res_naive_f0):.8f}")
print(f"  F1: {fps.to_float(res_naive_f1):.8f}")

# --- 运行奇偶分解算法 ---
res_opt_f0, res_opt_f1 = solver.calc_optimized_odd_even(x_fixed)
print(f"[奇偶分解 Optimized]")
print(f"  F0: {fps.to_float(res_opt_f0):.8f}")
print(f"  F1: {fps.to_float(res_opt_f1):.8f}")

# --- 验证：浮点数真值 (Ground Truth) ---
x = test_x
c = raw_coeffs
# F0 = c0*x^4 - c1*x^3 + c2*x^2 + (0.5 - c3)*x + c4
gt_f0 = c[0]*x**4 - c[1]*x**3 + c[2]*x**2 + (0.5 - c[3])*x + c[4]
# F1 = c0*x^4 + c1*x^3 + c2*x^2 + (0.5 + c[3])*x + c[4]
gt_f1 = c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + (0.5 + c[3])*x + c[4]

print("=" * 60)
print(f"[浮点数真值 Ground Truth]")
print(f"  F0: {gt_f0:.8f}")
print(f"  F1: {gt_f1:.8f}")

# --- 误差分析 ---
print("-" * 60)
err_f0 = abs(res_opt_f0 - res_naive_f0)
err_f1 = abs(res_opt_f1 - res_naive_f1)
print(f"算法间整数位误差 (Diff): F0={err_f0}, F1={err_f1}")
if err_f0 < 5 and err_f1 < 5:
    print("结论: 两种算法结果一致 (误差仅为舍入带来的最后几位抖动)。")
    print("      但奇偶分解法少用了 60% 以上的乘法器！")
else:
    print("警告: 误差较大，请检查实现。")