#!/usr/bin/env python3
"""Write the byte-range schedule a real hybrid decoder would follow.

nn/online.py splits the token stream into B contiguous streams and trains them in
lockstep: block (step s, stream b) = tokens [b*per + s*T, b*per + (s+1)*T). Prediction
for step s needs training on all streams' steps < s, so a decoder must produce bytes
in the order s = 0.., b = 0..B-1 -- not in file order.

    python tools/schedule.py <file> <out.order> --B 24 --T 256 [--vocab 16384]

Output: u64 LE (start, end) byte ranges, for `xzip eval --order`.
Must use the same tokenizer settings as the NN run.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nn"))
from tokenizer import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("out")
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=16384)
    a = ap.parse_args()
    data = open(a.file, "rb").read()
    vocab = Vocab.build(data, a.vocab)
    toks = vocab.tokenize(data)[0]
    lens = np.array([1] * 256 + [len(w) for w in vocab.words], dtype=np.int64)[toks]
    off = np.concatenate([[0], np.cumsum(lens)])  # byte offset of token i
    assert off[-1] == len(data)
    ntok = len(toks)
    per = math.ceil(ntok / a.B)
    steps = math.ceil(per / a.T)
    ranges = []
    for s in range(steps):
        for b in range(a.B):
            t0 = b * per + s * a.T
            t1 = min(b * per + (s + 1) * a.T, (b + 1) * per, ntok)
            if t0 < t1:
                ranges.append((off[t0], off[t1]))
    arr = np.array(ranges, dtype="<u8")
    arr.tofile(a.out)
    covered = int((arr[:, 1] - arr[:, 0]).sum())
    print(f"{len(ranges)} ranges, {covered} bytes, mean {covered / len(ranges):.0f} B/range")


if __name__ == "__main__":
    main()
