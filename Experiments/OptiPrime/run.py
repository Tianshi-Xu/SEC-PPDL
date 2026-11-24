#!/usr/bin/env python3
"""Two-party runner for the opt-in quantization/preprocessing experiment."""

import argparse
import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
from imagenet.bundle import read_arithmetic

dispatch = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
dispatch.add_argument("--kind", choices=["ot", "protocols", "he", "imagenet"])
entry, _ = dispatch.parse_known_args()
kind = entry.kind or "imagenet"
imagenet = kind == "imagenet"
parser = argparse.ArgumentParser(
    allow_abbrev=False,
    description=(
        "Run two-party inference on an exported ImageNet bundle."
        if imagenet
        else "Run two-party protocol checks."
    ),
)
parser.add_argument(
    "--kind",
    choices=["ot", "protocols", "he", "imagenet"],
    default=kind,
    help=argparse.SUPPRESS if imagenet else "Protocol check to run",
)
if imagenet:
    parser.add_argument(
        "--bundle", required=True, help="Exported graph, weights and public test images"
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=0,
        help="Open intermediate values for this many test images",
    )
parser.add_argument("--binary", help="Override the executable in build or build-gpu")
parser.add_argument(
    "--ot", choices=["iknp", "ferret"], help="OT backend"
)
parser.add_argument(
    "--preprocess",
    type=int,
    choices=[0, 1],
    help="Prepare OT offline (1) or use live OT (0)",
)
parser.add_argument("--threads", type=int, default=16)
modes = parser.add_mutually_exclusive_group()
modes.add_argument("--mode", choices=["exact", "approx"], default="exact")
modes.add_argument(
    "--approx-relu", action="store_const", dest="mode", const="approx", help=argparse.SUPPRESS
)
parser.add_argument("--linear", choices=["cheetah", "cpu", "gpu"])
parser.add_argument("--qbits", help="Comma-separated HE coefficient-prime bit lengths")
parser.add_argument(
    "--repeat", type=int, default=1, help="Number of images" if imagenet else "Repetitions"
)
parser.add_argument("--address", default="127.0.0.1")
parser.add_argument("--timeout", type=int)
parser.add_argument("--output", required=True)
if not imagenet:
    parser.add_argument("--accumulator-bits", type=int, choices=[20, 26, 32], default=32)
    parser.add_argument("--po2-bits", type=int, choices=[20, 26, 32], default=26)
    parser.add_argument("--parallel", type=int, choices=[0, 1], default=1)
    parser.add_argument("--legacy", type=int, choices=[0, 1], default=0)
    parser.add_argument("--plaintext-bits", type=int, default=26)
    parser.add_argument("--po2", type=int, choices=[0, 1], default=0)
args = parser.parse_args()
if args.repeat < 1 or not 1 <= args.threads <= 128:
    parser.error("repeat must be positive and threads must be in [1,128]")
if imagenet and not 0 <= args.trace_count <= args.repeat:
    parser.error("trace count must be between zero and repeat")
args.approx_relu = args.mode == "approx"
args.linear = args.linear or ("cpu" if imagenet else "gpu")
baseline = imagenet and args.linear == "cheetah"
args.ot = args.ot or ("ferret" if imagenet and not baseline else "iknp")
if args.preprocess is None:
    args.preprocess = int(imagenet and not baseline)
if args.timeout is None:
    args.timeout = max(1800, 30 * args.repeat) if imagenet else 1800
if imagenet:
    if baseline and (args.approx_relu or args.ot != "iknp" or args.preprocess or args.qbits):
        parser.error("Cheetah baseline uses exact mode, live IKNP and its original HE parameters")
    arith = read_arithmetic(args.bundle)
    work_bits, shift = arith.bits, arith.shift
    args.qbits = args.qbits or ("50,50" if work_bits == 20 else "50,50,50")
    primes = args.qbits.split(",")
    if len(primes) < 2 or any(not p.isdecimal() or not 2 <= int(p) <= 60 for p in primes):
        parser.error("--qbits requires at least two prime lengths between 2 and 60")
    if not args.binary:
        root = Path(__file__).resolve().parents[2]
        build = "build-gpu" if args.linear == "gpu" else "build"
        args.binary = str(root / build / "Experiments/OptiPrime/test_quant_imagenet")
elif not args.binary:
    parser.error("provide --binary for a benchmark or --bundle for ImageNet inference")
else:
    args.qbits = args.qbits or "44,44,44"
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").unlink(missing_ok=True)
try:
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
except (OSError, ValueError, IndexError):
    ephemeral_start = 32768
port_low = 11000
port_high = min(29000, ephemeral_start) - max(args.threads, 1) - 1
if port_high <= port_low:
    raise RuntimeError("No suitable benchmark port range below ephemeral ports")
probes = []
while True:
    try:
        port = port_low + secrets.randbelow(port_high - port_low)
        for i in range(max(args.threads, 1) + 1):
            probe = socket.socket()
            probes.append(probe)
            probe.bind(("127.0.0.1", port + i))
        break
    except OSError:
        for probe in probes:
            probe.close()
        probes = []
for probe in probes:
    probe.close()
files = []
children = []
session_start = time.monotonic()
paths = [out / f"party{p}.log" for p in (1, 2)]
try:
    for party, path in zip((1, 2), paths):
        command = [args.binary, f"r={party}", f"p={port}"]
        if args.kind == "imagenet":
            command += [
                f"bundle={Path(args.bundle).resolve()}",
                f"results={out.resolve()}",
                f"nt={args.threads}",
                f"count={args.repeat}",
                f"trace={args.trace_count}",
                f'gpu={int(args.linear=="gpu")}',
                f"approx_relu={int(args.approx_relu)}",
                f"qbits={args.qbits}",
                f"address={args.address}",
                f"cheetah={int(baseline)}",
            ]
            if not baseline and (args.ot != "ferret" or not args.preprocess):
                command.extend([f'ferret={int(args.ot == "ferret")}', f"prep={args.preprocess}"])
        elif args.kind != "he":
            command += [f'ot={int(args.ot=="ferret")}', f"prep={args.preprocess}"]
        if args.kind == "protocols":
            command += [
                f"nt={args.threads}",
                f"parallel={args.parallel}",
                f"legacy={args.legacy}",
                f"accbits={args.accumulator_bits}",
            ]
        if args.kind == "protocols":
            command += [
                f"po2={args.po2}",
                f"po2bits={args.po2_bits}",
                f"bits={args.plaintext_bits}",
            ]
        if args.kind == "he":
            command += [
                f'gpu={int(args.linear=="gpu")}',
                f"bits={args.plaintext_bits}",
                f"qbits={args.qbits}",
                f"po2={args.po2}",
            ]
        files.append(path.open("w"))
        children.append(subprocess.Popen(command, stdout=files[-1], stderr=subprocess.STDOUT))
    deadline = time.monotonic() + args.timeout
    while any(p.poll() is None for p in children):
        if any(p.poll() not in (None, 0) for p in children):
            raise RuntimeError(
                "Party return codes " + str([p.poll() for p in children]) + "; see " + str(out)
            )
        if time.monotonic() > deadline:
            raise TimeoutError("See " + str(out))
        time.sleep(0.1)
    if any(p.returncode for p in children):
        raise RuntimeError("A party failed; see " + str(out))
finally:
    for p in children:
        if p.poll() is None:
            p.kill()
        p.wait()
    for f in files:
        f.close()
session_seconds = time.monotonic() - session_start
if not imagenet and any(
    "PROTOCOL CHECK COMPLETE" not in path.read_text().splitlines() for path in paths
):
    raise RuntimeError("Missing two-party protocol check completion")
report = {"config": vars(args), "status": "pass"}
if args.kind == "imagenet":
    report["session_seconds"] = session_seconds
    report["config"] = {
        k: getattr(args, k)
        for k in (
            "kind",
            "linear",
            "approx_relu",
            "threads",
            "ot",
            "preprocess",
            "repeat",
            "trace_count",
        )
    }
    report["config"].update(
        bundle=str(Path(args.bundle).resolve()),
        mode=args.mode,
        work_ring_bits=60 if baseline else work_bits,
        shift=shift,
        weight_bits=arith.weight_bits,
        activation_bits=arith.activation_bits,
        fc_bits=arith.fc_bits,
        ciphertext_prime_bits="framework default" if baseline else [int(p) for p in primes],
    )
    parties = [
        [
            json.loads(line.removeprefix("INFERENCE "))
            for line in path.read_text().splitlines()
            if line.startswith("INFERENCE ")
        ]
        for path in paths
    ]
    if any(len(rows) != args.repeat for rows in parties):
        raise RuntimeError("Incomplete two-party timing records")
    report["runs"] = []
    for pair in zip(*parties):
        wall = max(pair, key=lambda r: r["online_seconds"])
        metrics = {
            k: sum(r[k] for r in pair) for k in ("linear_bytes", "offline_bytes", "online_bytes")
        }
        metrics.update(
            linear_seconds=wall["linear_seconds"],
            online_seconds=wall["online_seconds"],
            offline_seconds=max(p["offline_seconds"] for p in pair),
        )
        for unit in ("bytes", "seconds"):
            metrics["nonlinear_" + unit] = metrics["online_" + unit] - metrics["linear_" + unit]
        report["runs"].append(metrics)
    if len((out / "predictions.jsonl").read_text().splitlines()) != args.repeat:
        raise RuntimeError("Incomplete ImageNet predictions")
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    f"PASS {args.kind}: ot={args.ot} prep={args.preprocess} threads={args.threads} linear={args.linear}",
    flush=True,
)
