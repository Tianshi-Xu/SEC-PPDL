import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图风格
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

def main():
    df = pd.read_csv("silu_eval.csv")

    df["abs_err"] = (df["gt"] - df["he"]).abs()
    df["abs_err_masked"] = df["abs_err"]

    mae = df["abs_err_masked"].mean()
    max_err = df["abs_err_masked"].max()
    print(f"MAE (masked): {mae}")
    print(f"Max Err (masked): {max_err}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: GT vs HE
    ax1.plot(df["input"], df["gt"], label="Ground Truth", linewidth=2.5, color='royalblue', alpha=0.9)
    ax1.plot(df["input"], df["he"], label="HE/MPC Output", linestyle="--", linewidth=2, color='tomato', alpha=0.85)
    ax1.set_xlabel("Input", fontweight='bold')
    ax1.set_ylabel("Output", fontweight='bold')
    ax1.set_title("(a) SiLU: Ground Truth vs HE/MPC Output", fontweight='bold', pad=15)
    ax1.legend(frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图: Error
    ax2.plot(df["input"], df["abs_err_masked"], linewidth=2, color='orange', alpha=0.9)
    ax2.fill_between(df["input"], df["abs_err_masked"], alpha=0.3, color='orange')
    ax2.set_xlabel("Input", fontweight='bold')
    ax2.set_ylabel("Absolute Error", fontweight='bold')
    ax2.set_title(f"(b) Error Analysis (MAE: {mae:.6f}, Max: {max_err:.6f})", fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("gelu_eval.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    main()

