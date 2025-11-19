import math
import numpy as np
from torch import nn
import torch

import matplotlib.pyplot as plt



POLY_DEGREE = 4


def _fit_silu_poly(
    layer: nn.Module,
    point: float,
    degree: int = POLY_DEGREE,
    num_samples: int = 2000,
) -> np.ndarray:
    X = np.linspace(0.0, point, num_samples, dtype=np.float64)
    tensor_x = torch.tensor(X, dtype=torch.float64)
    Y = layer(tensor_x).detach().numpy()
    target = Y - 0.5 * X
    coefficients = np.polyfit(X, target, degree)
    return coefficients


def _evaluate_poly_float(x: np.ndarray, coefficients: np.ndarray, point: float) -> np.ndarray:
    abs_x = np.abs(x)
    y = np.polyval(coefficients, abs_x) + 0.5 * x
    y[x > point] = x[x > point]
    y[x < -point] = 0.0
    return y


def test_silu_acc():
    point = 4.6
    layer = nn.SiLU().double()
    coefficients = _fit_silu_poly(layer, point)
    print("coefficients:", coefficients)

    X = np.linspace(-15.0, 15.0, 1000, dtype=np.float64)
    tensor_x = torch.tensor(X, dtype=torch.float64)
    Y = layer(tensor_x).detach().numpy()
    y_smooth = _evaluate_poly_float(X, coefficients, point)
    print("mse of silu:", np.mean((Y - y_smooth) ** 2))

    if plt is None:
        print("matplotlib is not available; skipping plot generation.")
        return

    plt.plot(X, Y, label="silu")
    plt.plot(X, y_smooth, label="poly")
    plt.legend()
    plt.savefig("silu.png")
    plt.close()


def _quantize_to_fixed(x: torch.Tensor, frac_bits: int, total_bits: int) -> torch.Tensor:
    scale = 1 << frac_bits
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    x_scaled = torch.round(x * scale)
    x_clamped = torch.clamp(x_scaled, min_val, max_val)
    return x_clamped.to(torch.int64)


def _fixed_add(a: torch.Tensor, b: torch.Tensor, min_val: int, max_val: int) -> torch.Tensor:
    out = a + b
    return torch.clamp(out, min_val, max_val)


def _fixed_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    frac_bits: int,
    min_val: int,
    max_val: int,
) -> torch.Tensor:
    prod = a.to(torch.int64) * b.to(torch.int64)
    rounding = 1 << (frac_bits - 1)
    prod = torch.where(prod >= 0, prod + rounding, prod - rounding)
    prod = prod >> frac_bits
    return torch.clamp(prod, min_val, max_val)


def _estimate_int_bits(
    coefficients: np.ndarray, point: float, samples: int = 4096
) -> int:
    X = np.linspace(-point, point, samples, dtype=np.float64)
    Y = _evaluate_poly_float(X, coefficients, point)
    max_mag = max(1.0, np.max(np.abs(Y)))
    return max(3, math.ceil(math.log2(max_mag)) + 2)


def _evaluate_poly_fixed(
    x: torch.Tensor,
    coefficients: np.ndarray,
    point: float,
    frac_bits: int,
    int_bits: int,
) -> torch.Tensor:
    total_bits = 1 + int_bits + frac_bits
    scale = 1 << frac_bits
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    x_fp = _quantize_to_fixed(x, frac_bits, total_bits)
    point_fp = _quantize_to_fixed(torch.tensor(point, dtype=torch.float64), frac_bits, total_bits)
    inside = torch.abs(x_fp) <= point_fp
    center_x_fp = torch.where(inside, x_fp, torch.zeros_like(x_fp))
    abs_x_fp = torch.abs(center_x_fp)

    coeff_tensors = [
        _quantize_to_fixed(torch.tensor(float(val), dtype=torch.float64), frac_bits, total_bits)
        for val in coefficients
    ]
    half_fp = _quantize_to_fixed(torch.tensor(0.5, dtype=torch.float64), frac_bits, total_bits)

    # Horner evaluation for degree 4 polynomial in |x|
    y_fp = coeff_tensors[0]
    for coeff_fp in coeff_tensors[1:]:
        y_fp = _fixed_mul(y_fp, abs_x_fp, frac_bits, min_val, max_val)
        y_fp = _fixed_add(y_fp, coeff_fp, min_val, max_val)

    y_fp = _fixed_add(y_fp, _fixed_mul(half_fp, center_x_fp, frac_bits, min_val, max_val), min_val, max_val)

    y_fp = torch.where(inside, y_fp, torch.zeros_like(y_fp))
    y_fp = torch.where(x_fp > point_fp, x_fp, y_fp)
    y_fp = torch.where(x_fp < -point_fp, torch.zeros_like(y_fp), y_fp)

    return y_fp.to(torch.float64) / scale


def test_silu_bf16_vs_fixed_point():
    point = 4.6
    layer_fp64 = nn.SiLU().to(dtype=torch.float64)
    coefficients = _fit_silu_poly(layer_fp64, point)
    int_bits = _estimate_int_bits(coefficients, point)
    frac_bits_sweep = range(8, 25)

    x = torch.linspace(-point, point, 4096, dtype=torch.float64)
    y_ref = layer_fp64(x)

    layer_bf16 = nn.SiLU().to(dtype=torch.bfloat16)
    y_bf16 = layer_bf16(x.to(dtype=torch.bfloat16)).to(dtype=torch.float64)

    denom = torch.clamp(y_ref.abs(), min=1e-8)
    rel_err_bf16 = torch.abs((y_bf16 - y_ref) / denom)
    bf16_max_err = rel_err_bf16.max().item()
    bf16_mean_err = rel_err_bf16.mean().item()
    bf16_abs_diff = torch.abs(y_bf16 - y_ref)
    bf16_abs_max = bf16_abs_diff.max().item()
    bf16_abs_mean = bf16_abs_diff.mean().item()
    print(
        "bf16 relative error max={:.3e}, mean={:.3e}; absolute error max={:.3e}, mean={:.3e}".format(
            bf16_max_err, bf16_mean_err, bf16_abs_max, bf16_abs_mean
        )
    )

    target_max_err = 1e-2
    target_abs_err = 1e-2
    best_config = None
    first_abs_config = None  # track first config meeting absolute error target for plotting
    best_poly_curve = None
    first_abs_detail = None

    for frac_bits in frac_bits_sweep:
        total_bits = 1 + int_bits + frac_bits
        y_poly = _evaluate_poly_fixed(x, coefficients, point, frac_bits, int_bits)
        diff = y_poly - y_ref
        rel_err_poly = torch.abs(diff / denom)
        poly_rel_max_err = rel_err_poly.max().item()
        poly_mean_err = rel_err_poly.mean().item()
        abs_diff = torch.abs(diff)
        poly_abs_idx = torch.argmax(abs_diff).item()
        poly_abs_max = abs_diff[poly_abs_idx].item()
        poly_abs_mean = abs_diff.mean().item()
        print(
            "fixed-point total_bits={:2d} (int={}, frac={:2d}) -> rel_max={:.3e}, rel_mean={:.3e}, abs_max={:.3e}, abs_mean={:.3e}".format(
                total_bits, int_bits, frac_bits, poly_rel_max_err, poly_mean_err, poly_abs_max, poly_abs_mean
            )
        )
        if (
            best_config is None
            and poly_rel_max_err <= target_max_err
            and poly_abs_max <= target_abs_err
        ):
            best_config = (total_bits, frac_bits, poly_rel_max_err, poly_mean_err, poly_abs_max, poly_abs_mean)
            best_poly_curve = y_poly.detach().cpu().numpy()

        if first_abs_config is None and poly_abs_max <= target_abs_err:
            first_abs_config = (poly_abs_max, poly_abs_mean, total_bits, frac_bits)
            best_poly_curve = y_poly.detach().cpu().numpy()
            first_abs_detail = (
                x[poly_abs_idx].item(),
                y_ref[poly_abs_idx].item(),
                y_poly[poly_abs_idx].item(),
                poly_abs_max,
            )

    if best_config is not None:
        total_bits, frac_bits, max_err, mean_err, abs_err, abs_mean = best_config
        print(
            "first config meeting error targets: "
            f"total_bits={total_bits}, frac_bits={frac_bits}, rel_max={max_err:.3e}, "
            f"rel_mean={mean_err:.3e}, abs_max={abs_err:.3e}, abs_mean={abs_mean:.3e}"
        )
    else:
        print("No fixed-point config in sweep met the relative/absolute max-error targets.")

    if first_abs_detail is not None:
        x_val, ref_val, approx_val, err_val = first_abs_detail
        print(
            "first abs-target config point: x={:.4f}, float={:.6f}, fixed={:.6f}, abs_err={:.3e}".format(
                x_val, ref_val, approx_val, err_val
            )
        )

    if plt is not None and best_poly_curve is not None:
        x_np = x.detach().cpu().numpy()
        y_ref_np = y_ref.detach().cpu().numpy()
        y_bf16_np = y_bf16.detach().cpu().numpy()
        plt.figure(figsize=(8, 5))
        plt.plot(x_np, y_ref_np, label="float SiLU")
        plt.plot(x_np, y_bf16_np, label="bf16 SiLU", linestyle="--")
        label_poly = "fixed poly"
        if first_abs_config is not None:
            label_poly = (
                f"fixed poly (bits={first_abs_config[2]}, frac={first_abs_config[3]})"
            )
        plt.plot(x_np, best_poly_curve, label=label_poly, linestyle=":")
        plt.title("SiLU approximations")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("silu_compare.png")
        plt.close()
        print("Saved plot to silu_compare.png")


if __name__ == "__main__":
    # test_silu_acc()
    test_silu_bf16_vs_fixed_point()
