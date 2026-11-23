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
from tokenizer import LETTER, Vocab  # noqa: E402

CHUNK = 1 << 22


def vocab_words(data, size, maxlen=24):
    """Vocab.build's word list, counted in chunks (no whole-file Python lists)."""
    a = np.frombuffer(data, dtype=np.uint8)
    starts, ends = runs(a)
    counts = {}
    mv = memoryview(data)
    for c in range(0, len(starts), CHUNK):
        for s, e in zip(starts[c:c + CHUNK].tolist(), ends[c:c + CHUNK].tolist()):
            if 2 <= e - s <= maxlen:
                w = bytes(mv[s:e])
                counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1] * (len(kv[0]) - 1))
    return [w for w, c in ranked[: max(0, size - 256)] if c >= 2], starts, ends


def runs(a):
    isl = LETTER[a]
    starts = np.flatnonzero(isl[1:] & ~isl[:-1]) + 1
    if len(a) and isl[0]:
        starts = np.concatenate([[0], starts])
    ends = np.flatnonzero(isl[:-1] & ~isl[1:]) + 1
    if len(a) and isl[-1]:
        ends = np.concatenate([ends, [len(a)]])
    return starts, ends


def token_starts(data, size):
    """Cumulative token count per byte (uint32), matching Vocab.tokenize: an in-vocab
    letter run is one token, every other byte is its own token."""
    words, starts, ends = vocab_words(data, size)
    vocab = set(words)
    mv = memoryview(data)
    inner = np.zeros(len(data) + 1, dtype=np.int8)  # +1 at s+1, -1 at e for in-vocab runs
    for c in range(0, len(starts), CHUNK):
        st, en = starts[c:c + CHUNK], ends[c:c + CHUNK]
        hit = np.fromiter((bytes(mv[s:e]) in vocab for s, e in zip(st.tolist(), en.tolist())), bool, len(st))
        inner[st[hit] + 1] = 1
        inner[en[hit]] = -1
    del starts, ends
    np.cumsum(inner, out=inner)
    is_start = inner[:-1] == 0
    del inner
    return np.cumsum(is_start, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("out")
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--slow", action="store_true", help="reference path via Vocab.tokenize (small files only)")
    a = ap.parse_args()
    data = open(a.file, "rb").read()
    if a.slow:
        vocab = Vocab.build(data, a.vocab)
        toks = vocab.tokenize(data)[0]
        lens = np.array([1] * 256 + [len(w) for w in vocab.words], dtype=np.int64)[toks]
        offs = np.concatenate([[0], np.cumsum(lens)])  # byte offset of token i
        assert offs[-1] == len(data)
        ntok = len(toks)
        off = lambda t: offs[t]  # noqa: E731
    else:
        cs = token_starts(data, a.vocab)
        ntok = int(cs[-1])
        # byte offset of token t: first byte whose cumulative token count reaches t+1
        off = lambda t: np.where(t >= ntok, len(data), np.searchsorted(cs, t + 1))  # noqa: E731
    per = math.ceil(ntok / a.B)
    steps = math.ceil(per / a.T)
    s = np.repeat(np.arange(steps, dtype=np.int64), a.B)
    b = np.tile(np.arange(a.B, dtype=np.int64), steps)
    t0 = b * per + s * a.T
    t1 = np.minimum(np.minimum(t0 + a.T, (b + 1) * per), ntok)
    keep = t0 < t1
    arr = np.stack([off(t0[keep]), off(t1[keep])], axis=1).astype("<u8")
    arr.tofile(a.out)
    covered = int((arr[:, 1] - arr[:, 0]).sum())
    print(f"{len(arr)} ranges, {covered} bytes, mean {covered / len(arr):.0f} B/range")


if __name__ == "__main__":
    main()
