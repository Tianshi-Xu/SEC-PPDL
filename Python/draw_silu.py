import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 拟合逻辑 (反向工程还原系数来源)
# ==========================================
def fit_silu_coefficients_reverse_engineered(bound=4.6, num_points=10000):
    """
    演示如何通过拟合 SiLU(x) - 0.5x 来获取系数。
    这部分代码主要用于展示原理，实际近似函数中使用的是用户提供的固定系数。
    """
    print("正在执行反向拟合测试 (仅供验证原理)...")
    # 利用偶函数性质，只拟合正半轴 [0, bound]
    X = np.linspace(0, bound, num_points)
    # 目标: 拟合 SiLU(x) - 0.5x
    Y_true = X / (1 + np.exp(-X))
    Y_target = Y_true - 0.5 * X
    
    # 构建特征: [x^4, x^3, x^2, x^1, 1]
    features = np.vstack([X**4, X**3, X**2, X, np.ones_like(X)]).T
    
    # 线性回归拟合
    model = LinearRegression(fit_intercept=False)
    model.fit(features, Y_target)
    coeffs = model.coef_
    print(f"拟合完成。计算出的系数 (a,b,c,d,e): {coeffs}")
    print("-" * 30)
    return coeffs

# ==========================================
# 2. SiLU 函数定义 (原版 与 近似版)
# ==========================================

# 用户提供的精确系数 (硬编码以保证一致性)
COEFFS_USER = {
    'a': 0.0052629,
    'b': -0.06949947,
    'c': 0.32063845,
    'd': -0.02623376,
    'e': 0.00206111
}
BOUND_X = 4.6

def silu_original(x):
    """原版 SiLU 函数"""
    # 使用 np.where 处理大负数以防止 exp 溢出，虽然对于绘图范围不一定必要，但属良好实践
    return np.where(x < -20, 0.0, x / (1 + np.exp(-x)))

def silu_approx_poly(x):
    """
    分段多项式近似 SiLU.
    Formula:
        x > 4.6:  x
        x < -4.6: 0
        |x| <= 4.6: (a|x|^4 + b|x|^3 + c|x|^2 + d|x| + e) + 0.5x
    """
    a, b, c, d, e = COEFFS_USER['a'], COEFFS_USER['b'], COEFFS_USER['c'], COEFFS_USER['d'], COEFFS_USER['e']
    
    x_abs = np.abs(x)
    
    # 计算多项式部分 P(|x|)
    poly_even = (a * x_abs**4 + 
                 b * x_abs**3 + 
                 c * x_abs**2 + 
                 d * x_abs + 
                 e)
    
    # 核心近似: P(|x|) + 0.5x
    inner_val = poly_even + 0.5 * x
    
    # 分段逻辑
    result = np.where(x > BOUND_X, x, 
                      np.where(x < -BOUND_X, 0.0, inner_val))
    return result

# ==========================================
# 3. 主程序：执行与绘图
# ==========================================
def main():
    # (可选) 运行一下拟合函数看看结果是否接近
    # fit_silu_coefficients_reverse_engineered()

    # 准备绘图数据
    # 范围取比 [-4.6, 4.6] 稍大，以展示分段点外的行为
    x_plot = np.linspace(-6, 6, 2000)
    
    y_true = silu_original(x_plot)
    y_approx = silu_approx_poly(x_plot)
    
    # 计算绝对误差
    abs_error = np.abs(y_true - y_approx)
    max_error = np.max(abs_error)
    print(f"在测试区间 [-6, 6] 内的最大绝对误差: {max_error:.6e}")

    # 开始绘图
    plt.figure(figsize=(12, 10)) # 设置画布大小

    # --- 子图 1: 函数曲线对比 ---
    plt.subplot(2, 1, 1) # 2行1列，第1个图
    plt.title("SiLU: Original vs Piecewise Polynomial Approximation")
    
    # 绘制曲线
    plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='Original SiLU', alpha=0.6)
    plt.plot(x_plot, y_approx, 'r--', linewidth=1.5, label='Poly Approx (Order 4)')
    
    # 标记分段边界
    plt.axvline(x=BOUND_X, color='b', linestyle=':', label=f'Boundaries (±{BOUND_X})')
    plt.axvline(x=-BOUND_X, color='b', linestyle=':')
    
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylabel("Output value", fontsize=12)
    plt.xlabel("Input x", fontsize=12)

    # --- 子图 2: 绝对误差分析 ---
    plt.subplot(2, 1, 2) # 2行1列，第2个图
    plt.title(f"Absolute Error (|Original - Approx|)\nMax Error ≈ {max_error:.1e}")
    
    # 绘制误差曲线
    plt.plot(x_plot, abs_error, 'm-', linewidth=1.5, label='Absolute Error')
    
    # 标记分段边界区域
    plt.axvspan(-BOUND_X, BOUND_X, color='gray', alpha=0.1, label=f'Fitting Range [±{BOUND_X}]')
    plt.axvline(x=BOUND_X, color='b', linestyle=':')
    plt.axvline(x=-BOUND_X, color='b', linestyle=':')

    # 设置坐标轴
    # 如果误差非常小，可以尝试开启对数坐标: plt.yscale('log')
    # 这里线性坐标更能看出多项式拟合的波浪形误差特征
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.xlabel("Input x", fontsize=12)
    # 限制一下y轴范围，让误差波形更明显
    plt.ylim(0, max(0.001, max_error * 1.1)) 

    plt.tight_layout() # 调整布局防止重叠
    plt.savefig("silu_approximation.png")

if __name__ == "__main__":
    main()