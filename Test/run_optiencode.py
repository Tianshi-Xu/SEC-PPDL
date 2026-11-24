#!/usr/bin/env python3
"""Run both parties; fail on either error or a timeout, retain bounded logs."""
import argparse
from pathlib import Path
import socket
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
parser.add_argument('--face', action='store_true')
parser.add_argument('--mm', type=int, default=0)
parser.add_argument('--full', type=int, default=1)
parser.add_argument('--opti', type=int, default=1)
parser.add_argument('--threads', default='1,4')
parser.add_argument('--bits', type=int, default=40)
parser.add_argument('--degree', type=int, default=8192)
parser.add_argument('--repeat', type=int, default=2)
parser.add_argument('--bench', type=int, default=0)
parser.add_argument('--default-keys', type=int, default=0)
parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
parser.add_argument('--custom', type=int, choices=[0, 1], default=0)
parser.add_argument('--timeout', type=int, default=300)
parser.add_argument('--logs', default='build/test-results/optiencode')
args = parser.parse_args()
logs = Path(args.logs)
logs.mkdir(parents=True, exist_ok=True)
for threads in map(int, args.threads.split(',')):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    common = [args.binary, f'p={port}', f'nt={threads}', f'bits={args.bits}',
              f'n={args.degree}', f'repeat={args.repeat}', f'bench={args.bench}',
              f'default_keys={args.default_keys}', f'gpu={args.gpu}', f'custom={args.custom}']
    if args.face:
        common = [args.binary, f'p={port}', f'he_threads={threads}', f'full={args.full}', f'opti={args.opti}', f'mm={args.mm}']
    files, processes = [], []
    try:
        for party in (1, 2):
            label = f'face-opti{args.opti}-mm{args.mm}-full{args.full}' if args.face else f'bits{args.bits}-keys{args.default_keys}-gpu{args.gpu}-custom{args.custom}'
            path = logs / f'{label}-nt{threads}-party{party}.log'
            files.append(path.open('w'))
            processes.append(subprocess.Popen(common + [f'r={party}'], stdout=files[-1], stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('a party failed; see ' + str(logs))
            if time.monotonic() > deadline:
                raise TimeoutError('two-party test timed out')
            time.sleep(0.05)
        if any(p.returncode for p in processes):
            raise RuntimeError('a party failed; see ' + str(logs))
        marker = 'FaceResNet64 checks passed' if args.face else 'ALL OPTIENCODE CONVOLUTION CHECKS PASSED'
        for log in files:
            if marker not in Path(log.name).read_text():
                raise RuntimeError(f'Missing completion marker in {log.name}')
        print(f'PASS two-party test: {label}, threads={threads}', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for file in files:
            file.close()
