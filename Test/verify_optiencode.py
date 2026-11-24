#!/usr/bin/env python3
"""Plaintext-ring OptiEncode reference; Python 3.8+, no dependencies.
Source: https://arxiv.org/html/2609.16898v1 section III.
This checks algebra, not BFV noise/security or the C++ implementation.
"""
import random

CHECKS = 0


def check(a, b, label):
    global CHECKS
    CHECKS += 1
    if a != b:
        raise AssertionError(label)


class Ring:
    def __init__(self, n, t):
        if n < 2 or n & (n - 1) or t < 2:
            raise ValueError('invalid ring')
        self.n, self.t = n, t

    def zero(self):
        return [0] * self.n

    def term(self, z, exponent, value):
        q, i = divmod(exponent, self.n)
        z[i] = (z[i] + (-value if q % 2 else value)) % self.t

    def add(self, a, b):
        return [(x + y) % self.t for x, y in zip(a, b)]

    def shift(self, a, exponent):
        z = self.zero()
        for i, value in enumerate(a):
            self.term(z, i + exponent, value)
        return z

    def mul(self, a, b):
        z = self.zero()
        aa = [(i, x) for i, x in enumerate(a) if x]
        bb = [(j, y) for j, y in enumerate(b) if y]
        for i, x in aa:
            for j, y in bb:
                self.term(z, i + j, x * y)
        return z

    def auto(self, a, k):
        if k % 2 == 0:
            raise ValueError('Galois exponent must be odd')
        z = self.zero()
        for i, value in enumerate(a):
            self.term(z, i * k, value)
        return z


def trace(r, a, c):
    if c < 1 or c & (c - 1) or c > r.n:
        raise ValueError('require power-of-two C <= N')
    for j in range(c.bit_length() - 1):
        a = r.add(a, r.auto(a, r.n // (1 << j) + 1))
    return a


def sigma(r, a, c, power):
    # Sufficient condition for the paper's exact generator-power identities.
    if c < 1 or c & (c - 1) or c * c > 2 * r.n:
        raise ValueError('require power-of-two C and C^2 <= 2N')
    base = 1 if c == 1 else 1 + 2 * r.n // c
    return r.auto(a, pow(base, power, 2 * r.n))


def evaluate(r, inputs, weights, c, baby):
    if baby < 1 or c % baby:
        raise ValueError('baby must divide C')
    naive, direct, fast = r.zero(), r.zero(), r.zero()
    for x, ws in zip(inputs, weights):
        qs = []
        for o, w in enumerate(ws):
            naive = r.add(naive, r.shift(trace(r, r.mul(x, w), c), o))
        for j in range(c):
            q = r.zero()
            for o, w in enumerate(ws):
                q = r.add(q, r.shift(sigma(r, w, c, j), o))
            qs.append(q)
            direct = r.add(direct, r.mul(sigma(r, x, c, j), q))
        for b in range(0, c, baby):
            subtotal = r.zero()
            for a in range(baby):
                adjusted = sigma(r, qs[b + a], c, -b)
                subtotal = r.add(subtotal, r.mul(sigma(r, x, c, a), adjusted))
            fast = r.add(fast, sigma(r, subtotal, c, b))
    check(direct, naive, 'reformulation')
    check(fast, naive, 'BSGS')
    return naive


def encode(r, x, weights, c, padding):
    """Cross-correlation layout with channel tiling and zero-padded tails.
    input exponent = C*((h+pad)*Wp+w+pad)+channel
    weight exponent = -C*(kernel_h*Wp+kernel_w)-channel
    This explicit layout is not a literal copy of the paper's subscripts.
    Spatial tiling is not implemented.
    """
    ci, h, width = len(x), len(x[0]), len(x[0][0])
    wp = width + 2 * padding
    if c * (h + 2 * padding) * wp > r.n:
        raise ValueError('spatial tile exceeds ring capacity')
    kh, kw = len(weights[0][0]), len(weights[0][0][0])
    inputs, tiles = [], []
    for start in range(0, ci, c):
        inp, ws = r.zero(), []
        for ch in range(min(c, ci - start)):
            for i in range(h):
                for j in range(width):
                    index = c * ((i + padding) * wp + j + padding) + ch
                    inp[index] = x[start + ch][i][j] % r.t
        for weight in weights:
            poly = r.zero()
            for ch in range(min(c, ci - start)):
                for i in range(kh):
                    for j in range(kw):
                        r.term(poly, -c * (i * wp + j) - ch,
                               weight[start + ch][i][j])
            ws.append(poly)
        inputs.append(inp)
        tiles.append(ws)
    return inputs, tiles, wp


def main():
    rng = random.Random(17)
    # Include the modulus actually selected in OptiPrime CPU/main.cpp.
    for t in (65537, 0x1b4001, 1 << 21):
        r = Ring(128, t)
        a = [rng.randrange(t) for _ in range(r.n)]
        for c in (1, 2, 4, 8, 16):
            expected = [c * v % t if i % c == 0 else 0
                        for i, v in enumerate(a)]
            check(trace(r, a, c), expected, 'projection scale')
            total = r.zero()
            for j in range(c):
                total = r.add(total, sigma(r, a, c, j))
            check(total, expected, 'cyclic trace')
        r = Ring(32, t)
        for _ in range(8):
            a, b = [[rng.randrange(t) for _ in range(r.n)] for _ in range(2)]
            k = rng.randrange(1, 2 * r.n, 2)
            check(r.auto(r.mul(a, b), k),
                  r.mul(r.auto(a, k), r.auto(b, k)), 'homomorphism')
            check(r.auto(r.auto(a, k), pow(k, -1, 2 * r.n)), a, 'inverse')
        xs = [[rng.randrange(t) for _ in range(32)] for _ in range(2)]
        ws = [[[rng.randrange(t) for _ in range(32)] for _ in range(4)]
              for _ in xs]
        for baby in (1, 2, 4):
            evaluate(r, xs, ws, 4, baby)
        # Ci, Co, H, W, Kh, Kw, stride, padding, C.
        cases = [(4, 4, 1, 3, 1, 2, 1, 0, 4),
                 (3, 5, 3, 4, 2, 3, 1, 1, 4),
                 (9, 7, 4, 5, 3, 2, 2, 1, 4),
                 (5, 3, 3, 3, 1, 1, 1, 0, 2)]
        for ci, co, h, width, kh, kw, stride, pad, c in cases:
            x = [[[rng.randint(-4, 4) for _ in range(width)]
                  for _ in range(h)] for _ in range(ci)]
            weights = [[[[rng.randint(-3, 3) for _ in range(kw)]
                         for _ in range(kh)] for _ in range(ci)]
                       for _ in range(co)]
            oh = (h + 2 * pad - kh) // stride + 1
            ow = (width + 2 * pad - kw) // stride + 1
            for start in range(0, co, c):
                group = weights[start:start + c]
                r = Ring(256, t)
                xs, ws, wp = encode(r, x, group, c, pad)
                for j in range(c.bit_length()):
                    packed = evaluate(r, xs, ws, c, 1 << j)
                # Algebra only: recovery in modulus C*t before sharing/masking.
                lift = Ring(256, c * t)
                lx, lw, _ = encode(lift, x, group, c, pad)
                lifted = evaluate(lift, lx, lw, c, 1)
                for o, weight in enumerate(group):
                    for i in range(oh):
                        for j in range(ow):
                            expected = 0
                            for ch in range(ci):
                                for a in range(kh):
                                    for b in range(kw):
                                        u, v = i * stride + a - pad, j * stride + b - pad
                                        if 0 <= u < h and 0 <= v < width:
                                            expected += x[ch][u][v] * weight[ch][a][b]
                            index = c * (i * stride * wp + j * stride) + o
                            check(packed[index], c * expected % t, 'scalar convolution')
                            check(lifted[index] // c % t, expected % t, 'lifted recovery')
                            if t % 2:
                                check(packed[index] * pow(c, -1, t) % t,
                                      expected % t, 'odd-modulus recovery')
        print('PASS: modulus', t)
    check(4 * 0 % (1 << 21), 4 * (1 << 19) % (1 << 21), 'scale loses information')
    check(pow(9, 2, 128), 81, 'generator counterexample')
    if pow(9, 2, 128) == 17:
        raise AssertionError('generator counterexample failed')
    print('PASS:', CHECKS, 'assertions; BFV/C++ backend not tested')


if __name__ == '__main__':
    main()
