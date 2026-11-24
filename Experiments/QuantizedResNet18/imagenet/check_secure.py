"""Replay opened public test traces with independently computed integer bounds."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from model import IntegerModel, ReleasedModel
from bundle import checkpoint, read_arithmetic


def word(stream):
    return struct.unpack("<I", stream.read(4))[0]


@torch.inference_mode()
def check(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Compare in double precision, with exact integer dot products on these inputs.
    config = json.loads((args.results / "summary.json").read_text())["config"]
    approximate = config["approx_relu"]
    bundle = Path(config["bundle"])
    arch, source = checkpoint(bundle)
    arith = read_arithmetic(bundle)
    model = IntegerModel(
        ReleasedModel(args.assets, args.device, arch, source),
        json.loads((bundle / "scales.json").read_text()),
        *arith,
    )
    rows = [
        json.loads(line) for line in (args.results / "predictions.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((bundle / "cases.json").read_text())
    if len(rows) != config["repeat"]:
        raise ValueError("Incomplete predictions")
    report = dict(
        images=len(rows),
        traces=0,
        activation_values_checked=0,
        layout_steps_checked=0,
        logits_replayed=0,
        nonintegral_values=0,
        rounded_up=0,
        correct_top1=0,
        correct_top5=0,
        exact_top1_agreement=0,
        mode="approx" if approximate else "exact",
    )
    stats = torch.zeros(4, dtype=torch.int64, device=args.device)
    with (bundle / "cases.bin").open("rb") as cases:
        wide = arith.activation_bits > 8
        if cases.read(4) != (b"QIC2" if wide else b"QIC1") or word(cases) != len(manifest):
            raise ValueError("Invalid inputs")
        for i, row in enumerate(rows):
            y = word(cases)
            data = np.frombuffer(
                cases.read(3 * 224 * 224 * (2 if wide else 1)), dtype="<u2" if wide else "u1"
            ).copy()
            expected = np.frombuffer(cases.read(4000), dtype="<i4")
            actual = np.array(row["logits"])
            if (
                i != row["index"]
                or y != row["label"]
                or y != manifest[i][1]
                or actual.shape != (1000,)
                or row["winner"] != actual.argmax()
            ):
                raise ValueError(f"Output or label mismatch for image {i}")
            report["correct_top1"] += int(row["winner"] == y)
            report["correct_top5"] += int(y in np.argsort(-actual, kind="stable")[:5])
            report["exact_top1_agreement"] += int(row["winner"] == expected.argmax())
            if not approximate and not np.array_equal(actual, expected):
                raise ValueError(f"Exact mode disagrees at image {i}")
            if i >= config["trace_count"]:
                continue
            with (args.results / f"trace-{i}.bin").open("rb") as trace:
                full_trace = trace.read(4) == b"QTR2"
                if not full_trace:
                    trace.seek(0)

                def observe(name, lower, upper):
                    n = word(trace)
                    if n != lower.numel():
                        raise ValueError(f"Trace shape mismatch at {i}/{name}")
                    raw = np.frombuffer(trace.read(4 * n), dtype="<i4").copy()
                    opened = torch.from_numpy(raw).to(args.device).reshape(lower.shape)
                    bad = (opened < lower) | (opened > upper)
                    if bad.any():
                        raise ValueError(f"{i}/{name}: {int(bad.sum())} values outside bounds")
                    report["activation_values_checked"] += n
                    report["nonintegral_values"] += int((lower != upper).sum())
                    report["rounded_up"] += int(((lower != upper) & (opened == upper)).sum())
                    return opened

                x = torch.from_numpy(data).to(args.device, torch.float64).reshape(1, 3, 224, 224)

                def check_layout(name, value):
                    observe(name, value, value)
                    report["layout_steps_checked"] += 1

                replayed = (
                    model(
                        x,
                        approximate,
                        oracle=observe,
                        stats=stats,
                        observe_layout=check_layout if full_trace else None,
                    )
                    .cpu()
                    .numpy()[0]
                )
                if trace.read(1) or not np.array_equal(actual, replayed):
                    raise ValueError(f"Trailing trace or replayed FC logits disagree at image {i}")
                report["traces"] += 1
                report["logits_replayed"] += actual.size
    report["traced_scaled_numerator_wraps"] = int(stats[0])
    report["traced_fc_dot_wraps"] = int(stats[3])
    (args.results / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    check(parser.parse_args())
