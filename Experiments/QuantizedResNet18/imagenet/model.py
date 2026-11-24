"""Trained ImageNet models and their two integer inference modes."""

import json
import math
import struct
from collections import defaultdict
from statistics import median

import torch
import torch.nn.functional as F
from architecture import BLOCKS, forward, topology
from bundle import Arithmetic


def input_byte(x, activation_bits=8):
    midpoint = 1 << (activation_bits - 1)
    return (x.mul(46 * midpoint / 128).round() + midpoint).clamp(0, 2 * midpoint - 1)


def linear(step, x, w, b=None):
    if w.ndim == 2:
        return F.linear(x.flatten(1), w, b)[:, :, None, None]
    return F.conv2d(x, w, b, step.shape[4], step.shape[5])


class ReleasedModel:
    def __init__(self, assets, device, arch="resnet18", source=None):
        self.arch = arch
        self.source = source or ("aimet" if arch == "resnet18" else "torchvision")
        self.steps = topology(arch)
        self.final = self.steps[-1].name
        self.enc = {}
        if self.source == "aimet":
            if arch != "resnet18":
                raise ValueError("The AIMET checkpoint is for ResNet-18")
            self.state = torch.load(
                assets / "resnet18_w4a8_state_dict.pth", map_location=device, weights_only=True
            )
            self.enc = json.loads((assets / "resnet18_W4A8.encodings").read_text())
        else:
            from torchvision.models import get_model, get_model_weights

            weights = get_model_weights(arch).IMAGENET1K_V1
            state = torch.hub.load_state_dict_from_url(
                weights.url,
                model_dir=str(assets / "torchvision"),
                map_location="cpu",
                progress=False,
                check_hash=True,
                weights_only=True,
            )
            net = get_model(arch, weights=None).eval()
            net.load_state_dict(state)
            self.state = {}
            for s in self.steps:
                if s.op != "conv":
                    continue
                layer = net.get_submodule(s.name)
                w = layer.weight.detach()
                b = layer.bias.detach() if layer.bias is not None else torch.zeros(w.shape[0])
                if arch.startswith("resnet") and s.name != "fc":
                    bn_name = (
                        s.name.replace("downsample.0", "downsample.1")
                        if "downsample" in s.name
                        else s.name.replace("conv", "bn")
                    )
                    bn = net.get_submodule(bn_name)
                    factor = bn.weight.detach() / (bn.running_var + bn.eps).sqrt()
                    w = w * factor[:, None, None, None]
                    b = (b - bn.running_mean) * factor + bn.bias.detach()
                self.state[s.name + ".weight"] = w.to(device)
                self.state[s.name + ".bias"] = b.to(device)
            # Independently verify the public schedule, BN folding and classifier layout.
            sample = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224).to(device)
            ref = net.to(device)(sample).detach()
            torch.testing.assert_close(self(sample), ref, rtol=1e-3, atol=2e-4)

    def __call__(self, x, observe=None):
        def conv(s, a):
            y = linear(s, a, self.state[s.name + ".weight"], self.state[s.name + ".bias"])
            if s.name != self.final and not s.signed:
                y = y.relu()
            if observe and s.name != self.final:
                observe(s.name, y)
            return y

        def residual(s, a, b):
            y = (a + b).relu()
            if observe:
                observe(s.name, y)
            return y

        return forward(self.steps, x, conv, residual, lambda a: a.mean((2, 3), keepdim=True))

    def weight_scale(self, name, bits):
        w = self.state[name + ".weight"]
        enc = self.enc.get(name + ".weight")
        if enc and bits == enc[0]["bitwidth"]:
            return w.new_tensor([e["scale"] for e in enc])
        limit = (1 << (bits - 1)) - 1
        peak = w.abs().amax() if name == self.final else w.flatten(1).abs().amax(1)
        return (peak / limit).reshape(-1).clamp_min(torch.finfo(w.dtype).tiny)


def calibrate(model, loader, device, balance_gains=True, arith=Arithmetic()):
    samples = defaultdict(list)

    def observe(k, a):
        flat = a.flatten(1)
        samples[k].append(flat[:, :: max(1, flat.shape[1] // 2048)].abs().cpu().flatten())

    for images, _, _ in loader:
        model(images.to(device), observe)
    q = {k: max(float(torch.quantile(torch.cat(v), 0.9995)), 1e-8) for k, v in samples.items()}
    mid = 1 << (arith.activation_bits - 1)
    cap = 2 * mid - 1
    unit = {k: v / cap for k, v in q.items()}
    if model.arch.startswith("resnet"):
        for stage, n in enumerate(BLOCKS[model.arch], 1):
            prefix = f"layer{stage}."
            endpoints = [
                s for s in model.steps if s.name.startswith(prefix) and (s.signed or s.op == "add")
            ]
            stage_unit = max(q[s.name] / (mid - 1 if s.signed else cap) for s in endpoints)
            if stage == 1 and model.arch != "resnet50":
                stage_unit = max(stage_unit, unit["conv1"])
                unit["conv1"] = stage_unit
            for s in endpoints:
                unit[s.name] = stage_unit
    scales = {}
    units = [128 / (46 * mid)]
    for s in model.steps:
        incoming = units[s.src]
        outgoing = unit.get(s.name, incoming)
        if s.op == "conv":
            if (
                balance_gains
                and s.name.endswith(".conv1")
                and model.arch.startswith("resnet")
                and model.arch != "resnet50"
            ):
                after = s.name[:-1] + "2"
                a, b = (
                    median(model.weight_scale(k, arith.weight_bits).tolist())
                    for k in (s.name, after)
                )
                outgoing = max(outgoing, math.sqrt(incoming * a * unit[after] / b))
            scales[s.name] = (incoming, None if s.name == model.final else outgoing)
        units.append(outgoing)
    return scales


class IntegerModel:
    def __init__(
        self, release, scales, bits=20, shift=9, weight_bits=4, activation_bits=8, fc_bits=8
    ):
        self.arith = Arithmetic(bits, shift, weight_bits, activation_bits, fc_bits).validate()
        self.steps, self.final, self.arch = release.steps, release.final, release.arch
        self.source = release.source
        self.params = {}
        self.max_row_l1 = {}
        for s in self.steps:
            if s.op != "conv":
                continue
            k = s.name
            bw = fc_bits if k == self.final else weight_bits
            sw = release.weight_scale(k, bw)
            w, b = release.state[k + ".weight"], release.state[k + ".bias"]
            limit = 1 << (bw - 1)
            wq = (w / sw.view(-1, *([1] * (w.ndim - 1)))).round().clamp(-limit, limit - 1)
            self.max_row_l1[k] = int(wq.flatten(1).abs().sum(1, dtype=torch.float64).amax())
            sin, sout = scales[k]
            gain = None
            if k == self.final:
                bias = (b / (sin * sw[0])).round()
            else:
                gain = (sin * sw / sout * (1 << shift)).round().long()
                bias = (b / sout * (1 << shift)).round()
            if bias.abs().amax() >= 2**31 or (gain is not None and gain.max() >= 2**32):
                raise ValueError(f"Exported multiplier or bias exceeds bundle storage at {k}")
            self.params[k] = wq, bias.int(), gain

    def export(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(
            json.dumps(
                {"architecture": self.arch, "source": self.source, **self.arith._asdict()}, indent=2
            )
            + "\n"
        )
        mid = 1 << (self.arith.activation_bits - 1)
        with (folder / "meta.bin").open("wb") as meta, (folder / "weights.bin").open(
            "wb"
        ) as weights, (folder / "graph.txt").open("w") as graph:
            meta.write(struct.pack("<4s5I", b"QIM2", *self.arith))
            for s in self.steps:
                if s.op == "conv":
                    print(
                        s.op,
                        s.src,
                        *s.shape,
                        mid if s.src == 0 else 0,
                        mid if s.signed else 0,
                        file=graph,
                    )
                    w, b, m = self.params[s.name]
                    if m is not None:
                        meta.write(m.cpu().numpy().astype("<u4").tobytes())
                    weights.write(w.cpu().numpy().astype("i1").tobytes())
                    weights.write(b.cpu().numpy().astype("<i4").tobytes())
                elif s.op == "add":
                    print(s.op, s.src, s.skip, int(s.projection), file=graph)
                else:
                    print(s.op, s.src, file=graph)

    def __call__(
        self, image, approximate=False, generator=None, oracle=None, stats=None, observe_layout=None
    ):
        bits, shift, wb, ab, fb = self.arith
        mid, divisor = 1 << (ab - 1), 1 << shift
        cap = mid * 2 - 1

        def ring(v, width=bits):
            half = 1 << (width - 1)
            return (v + half).remainder(2 * half) - half

        def conv(s, a):
            w, b, m = self.params[s.name]
            if s.src == 0:
                a = a - mid
            # FP32 represents every integer partial sum up to 2^24. Use FP64
            # when the weight row norm and activation range exceed that bound.
            safe_input = (1 << 24) // max(1, self.max_row_l1[s.name])
            limit = (1 << (bits - 1)) if approximate else cap
            if a.dtype == torch.float32 and limit > safe_input and a.abs().amax() > safe_input:
                a = a.double()
            dot = linear(s, a, w.to(a.dtype)).round().long()
            if s.name == self.final:
                if stats is not None:
                    stats[3] += (dot != ring(dot)).sum()
                return ring(ring(dot) + b.view(1, -1, 1, 1), max(bits, 26))
            numerator = dot * m.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)
            z = mid if s.signed else 0
            if approximate:
                numerator = numerator + z * divisor
                y = ring(numerator).clamp_min(0)
                lo, hi = y >> shift, (y + divisor - 1) >> shift
                if oracle:
                    out = oracle(s.name, lo, hi)
                else:
                    coin = torch.randint(
                        divisor, y.shape, dtype=torch.int32, device=y.device, generator=generator
                    )
                    out = (y + coin) >> shift
            else:
                out = ((ring(numerator) >> shift) + z).clamp(0, cap)
                if oracle:
                    out = oracle(s.name, out, out)
            if stats is not None:
                stats[0] += (numerator != ring(numerator)).sum()
                stats[1] += (out > cap).sum()
            return out.to(a.dtype)

        def residual(s, a, b):
            y = ring(a.long() + b.long() - mid * (1 + int(s.projection))).clamp_min(0)
            if not approximate:
                y = y.clamp_max(cap)
            if oracle:
                y = oracle(s.name, y, y)
            if stats is not None:
                stats[1] += (y > cap).sum()
            return y.to(a.dtype)

        def mean(a):
            sums = a.long().sum((2, 3), keepdim=True)
            if stats is not None:
                stats[2] += (sums >= 1 << bits).sum()
            return (sums // 49).to(a.dtype)

        # Preserve exact integer sums for higher-precision W/A configurations.
        if max(wb, fb) + ab > 16 or ab > 8 or (approximate and bits - shift >= 24):
            image = image.double()
        return forward(self.steps, image, conv, residual, mean, observe_layout)
