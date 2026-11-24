"""Calibrate the released checkpoint, evaluate both graphs and export held-out MPC cases."""

import argparse
import csv
import hashlib
import json
import random
import struct
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import IntegerModel, ReleasedModel, calibrate, input_byte
from bundle import Arithmetic, checkpoint, read_arithmetic
from architecture import BLOCKS


class Images(Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label = self.entries[i]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, label, str(path)


def files(root, label):
    return [
        (p, label(p.parent.name))
        for p in sorted(root.glob("*/*"))
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith(".")
    ]


def loader(entries, batch):
    return DataLoader(Images(entries), batch_size=batch, num_workers=6, shuffle=False)


def validate_weights(root):
    expected = {
        "resnet18_w4a8_state_dict.pth": "55c60a57091ac9efeeaf4d4f807f798f33529e5eb03eb7a111939889110b0621",
        "resnet18_W4A8.encodings": "d1991962bc19d97deec69048d195fd51867b6e2d70715b635c5b0c0836cea032",
    }
    for k, v in expected.items():
        if hashlib.sha256((root / k).read_bytes()).hexdigest() != v:
            raise ValueError("Unexpected checkpoint or encoding digest: " + k)


def export_cases(folder, entries, model, device):
    with (folder / "cases.bin").open("wb") as f:
        wide = model.arith.activation_bits > 8
        f.write((b"QIC2" if wide else b"QIC1") + struct.pack("<I", len(entries)))
        for x, y, _ in loader(entries, 4):
            xb = input_byte(x.to(device), model.arith.activation_bits)
            # FP64 cross-check for the public audit vectors.
            target = model(xb.double()).cpu()
            if not torch.equal(target, model(xb).cpu()):
                raise ValueError("FP32 dot product did not match FP64 on audit vector")
            encoded = xb.cpu().numpy().astype("<u2" if wide else "u1")
            for image, label, ref in zip(encoded, y, target.numpy()):
                f.write(struct.pack("<I", int(label)))
                f.write(image.tobytes())
                f.write(ref.astype("<i4").tobytes())
    (folder / "cases.json").write_text(json.dumps([(str(p), y) for p, y in entries], indent=2))


def evaluate(name, entries, model, released, out, batch, device):
    rng = torch.Generator(device=device).manual_seed(20261124)
    totals = {m: [0, 0] for m in ("released", "exact", "approx")}
    stats = {m: torch.zeros(4, dtype=torch.int64, device=device) for m in ("exact", "approx")}
    agreement = lost = gained = 0
    with (out / (name + ".csv")).open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image",
                "label",
                "released",
                "exact",
                "approx",
                "released_top5",
                "exact_top5",
                "approx_top5",
            ]
        )
        for i, (x, y, paths) in enumerate(loader(entries, batch)):
            x = x.to(device)
            xb = input_byte(x, model.arith.activation_bits)
            scores = {
                "released": released(x),
                "exact": model(xb, stats=stats["exact"]),
                "approx": model(xb, True, rng, stats=stats["approx"]),
            }
            # Stable ordering gives the same first-index tie convention as the MPC argmax.
            ranks = {
                k: v.argsort(dim=1, descending=True, stable=True)[:, :5].cpu()
                for k, v in scores.items()
            }
            top = {k: v[:, 0] for k, v in ranks.items()}
            top5 = {k: (v == y[:, None]).any(dim=1) for k, v in ranks.items()}
            for k in scores:
                totals[k][0] += int((top[k] == y).sum())
                totals[k][1] += int(top5[k].sum())
            agreement += int((top["exact"] == top["approx"]).sum())
            lost += int(((top["exact"] == y) & (top["approx"] != y)).sum())
            gained += int(((top["exact"] != y) & (top["approx"] == y)).sum())
            writer.writerows(
                zip(
                    paths,
                    y.tolist(),
                    *(v.tolist() for v in top.values()),
                    *(v.tolist() for v in top5.values()),
                )
            )
            if i % 40 == 0:
                print(name, min((i + 1) * batch, len(entries)), totals, flush=True)
    report = dict(
        images=len(entries),
        classes=len({y for _, y in entries}),
        correct=totals,
        agreement=agreement,
        lost=lost,
        gained=gained,
        counts={
            k: dict(
                zip(
                    (
                        "scaled_numerator_wraps",
                        "activations_above_quantizer_max",
                        "pool_sums_requiring_widening",
                        "fc_dot_wraps",
                    ),
                    v.tolist(),
                )
            )
            for k, v in stats.items()
        },
    )
    (out / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(name, report, flush=True)


def repeat_rounding(name, entries, model, out, batch, device, seed):
    rng = torch.Generator(device=device).manual_seed(seed)
    top1 = top5 = 0
    for x, y, _ in loader(entries, batch):
        logits = model(input_byte(x.to(device), model.arith.activation_bits), True, rng)
        ranks = logits.argsort(dim=1, descending=True, stable=True)[:, :5].cpu()
        top1 += int((ranks[:, 0] == y).sum())
        top5 += int((ranks == y[:, None]).any(dim=1).sum())
    result = dict(images=len(entries), seed=seed, top1=top1, top5=top5)
    (out / f"{name}-seed-{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, result, flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--secure-images", type=int, default=32)
    parser.add_argument("--model", choices=BLOCKS, default=None)
    parser.add_argument("--weights-source", choices=("aimet", "torchvision"))
    parser.add_argument("--weight-bits", type=int, choices=range(2, 9))
    parser.add_argument("--activation-bits", type=int, choices=range(2, 17))
    parser.add_argument("--fc-bits", type=int, choices=range(2, 9))
    parser.add_argument(
        "--shift", type=int, help="Fractional bits of public convolution multipliers (default 9)"
    )
    parser.add_argument(
        "--work-bits",
        type=int,
        choices=range(20, 33),
        help="HE plaintext and MPC work-ring width (default 26 for VGG19, 20 otherwise)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export the bundle and selected test cases, then skip full-dataset scoring",
    )
    parser.add_argument(
        "--balance-gains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance each block's intermediate unit using its published weight scales",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--rounding-seeds",
        type=int,
        nargs="+",
        help="Repeat the approximate evaluation with the existing exported scales",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rounding_seeds:
        arch, source = checkpoint(args.output)
        if args.weights_source and args.weights_source != source:
            parser.error("--weights-source differs from the exported bundle")
        arith = read_arithmetic(args.output)
        if args.model and args.model != arch:
            parser.error("--model differs from the exported bundle")
    else:
        arch = args.model or "resnet18"
        source = args.weights_source or ("aimet" if arch == "resnet18" else "torchvision")
        arith = Arithmetic(
            args.work_bits or (26 if arch == "vgg19" else 20),
            9 if args.shift is None else args.shift,
            args.weight_bits or 4,
            args.activation_bits or 8,
            args.fc_bits or 8,
        ).validate()
    if source == "aimet":
        validate_weights(args.assets)
    indices = json.loads((args.data / "imagenet_class_index.json").read_text())
    index = {v[0]: int(k) for k, v in indices.items()}
    im = args.data / "imagenette2-320"
    train = files(im / "train", index.__getitem__)
    val = files(im / "val", index.__getitem__)
    v2 = files(args.data / "imagenetv2-matched-frequency-format-val", int)
    if (len(train), len(val), len(v2)) != (9469, 3925, 10000) or len({y for _, y in v2}) != 1000:
        raise ValueError(f"Incomplete datasets: {len(train)}, {len(val)}, {len(v2)}")
    released = ReleasedModel(args.assets, device, arch, source)
    if args.rounding_seeds:
        requested = (
            args.work_bits,
            args.shift,
            args.weight_bits,
            args.activation_bits,
            args.fc_bits,
        )
        if any(v is not None and v != stored for v, stored in zip(requested, arith)):
            parser.error("Requested bit widths differ from the exported bundle")
        model = IntegerModel(
            released, json.loads((args.output / "scales.json").read_text()), *arith
        )
        for seed in args.rounding_seeds:
            for name, entries in [("imagenetv2", v2), ("imagenette", val)]:
                repeat_rounding(name, entries, model, args.output, args.batch, device, seed)
        return
    calibration = random.Random(29).sample(train, 512)
    (args.output / "calibration.txt").write_text("\n".join(str(p) for p, _ in calibration) + "\n")
    scales = calibrate(released, loader(calibration, args.batch), device, args.balance_gains, arith)
    (args.output / "scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    model = IntegerModel(released, scales, *arith)
    model.export(args.output)
    selected = random.Random(117).sample(v2, args.secure_images)
    export_cases(args.output, selected, model, device)
    if args.export_only:
        return
    for name, entries in [("imagenetv2", v2), ("imagenette", val)]:
        evaluate(name, entries, model, released, args.output, args.batch, device)


if __name__ == "__main__":
    main()
