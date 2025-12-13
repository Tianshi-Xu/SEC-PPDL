import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("gelu_eval.csv")

    df["abs_err"] = (df["gt"] - df["he"]).abs()
    df_filtered = df.loc[~((df.index >= 2048) & (df.index < 4096)) & ~((df.index >= 6144) & (df.index < 8192))]
    mae = df_filtered["abs_err"].mean()
    max_err = df_filtered["abs_err"].max()
    print(f"MAE (filtered): {mae}")
    print(f"Max Err (filtered): {max_err}")

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


if __name__ == "__main__":
    main()

