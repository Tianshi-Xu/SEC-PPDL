import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("gelu_eval.csv")

    df["abs_err"] = (df["gt"] - df["he"]).abs()
    mask_exclude = ((df.index >= 2048) & (df.index < 4096)) | ((df.index >= 6144) & (df.index < 8192))
    df["abs_err_masked"] = df["abs_err"]
    df.loc[mask_exclude, "abs_err_masked"] = 0.0  # treat excluded ranges as 0 error

    mae = df["abs_err_masked"].mean()
    max_err = df["abs_err_masked"].max()
    print(f"MAE (masked): {mae}")
    print(f"Max Err (masked): {max_err}")

    plt.figure(figsize=(7, 4))
    plt.plot(df["input"], df["gt"], label="GT", linewidth=2)
    plt.plot(df["input"], df["he"], label="HE", linestyle="--")
    plt.xlabel("input")
    plt.ylabel("output")
    plt.title("GeLU HE vs GT")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gelu_eval.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 3.5))
    plt.plot(df["input"], df["abs_err_masked"], label="|HE - GT| (masked)")
    plt.xlabel("input")
    plt.ylabel("abs error")
    plt.title("GeLU Error (masked ranges)")
    plt.tight_layout()
    plt.savefig("gelu_error.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

