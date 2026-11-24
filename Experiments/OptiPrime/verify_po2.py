#!/usr/bin/env python3
"""Independent integer-ring checks for lifted PO2 OptiEncode and masked sharing."""
import importlib.util
from pathlib import Path
import random

path = Path(__file__).resolve().parents[2] / 'Test/verify_optiencode.py'
spec = importlib.util.spec_from_file_location('algebra', path)
algebra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algebra)
rng = random.Random(73)
checks = 0
for q in (65537, 65537 * 114689):
    ring = algebra.Ring(128, q)
    phase = [rng.randrange(q) for _ in range(128)]
    for channels in (1, 2, 4, 8, 16):
        inverse = pow(channels, -1, q)
        normalized = [v * inverse % q for v in phase]
        projected = algebra.trace(ring, normalized, channels)
        assert projected == [v if i % channels == 0 else 0 for i, v in enumerate(phase)]
        checks += len(phase)
print(f'PASS ciphertext modulus inverse + trace: {checks} coefficients')
for bits in (4, 10, 26):
    modulus, scale = 1 << bits, 16
    ring = algebra.Ring(128, modulus * scale)
    for channels in (1, 2, 4, 8, 16):
        for _ in range(3):
            # Additive shares are in the smaller ring, embedded without a
            # ring extension. Their wrap is annihilated by the output scale.
            clear = [rng.randrange(modulus) for _ in range(ring.n)]
            first = [rng.randrange(modulus) for _ in clear]
            second = [(x - a) % modulus for x, a in zip(clear, first)]
            lifted_input = [(a + b) % ring.t for a, b in zip(first, second)]
            weights = [[rng.randrange(-8, 8) for _ in clear] for _ in range(channels)]
            scaled = [[v * (scale // channels) % ring.t for v in row] for row in weights]
            output = algebra.evaluate(ring, [lifted_input], [scaled], channels,
                                      1 << ((channels.bit_length() - 1) // 2))
            # Expected also has the C trace factor, so compare against exact
            # integer trace over a ring large enough before reducing.
            exact_ring = algebra.Ring(128, modulus * scale)
            exact = algebra.evaluate(exact_ring, [clear], [weights], channels, 1)
            assert all(v % channels == 0 for v in exact)
            assert output == [(v // channels * scale) % ring.t for v in exact]
            assert all(v % scale == 0 for v in output)
            masks = [rng.randrange(modulus) for _ in output]
            masked = [(v - scale * m) % ring.t for v, m in zip(output, masks)]
            recovered = [(v // scale + m) % modulus for v, m in zip(masked, masks)]
            assert recovered == [v // channels % modulus for v in exact]
            checks += len(output)
print(f'PASS lifted PO2: {checks} coefficients; trace, BSGS, share wrap, subgroup masks')
