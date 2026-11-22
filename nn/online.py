"""Online-learned transformer over the word-token stream (research harness).

Reports the NN-only code length (bits/byte over the token stream) and writes a
side file of per-bit probabilities (N x 8 uint16, 0 = no prediction) that the
Rust CM mixes in (`xzip eval <file> --side <side>`).

NNCP-style: the token stream is cut into B contiguous streams processed in
lockstep. Each step codes the next T tokens of every stream with weights frozen
(context = previous T tokens), then takes `inner` optimizer steps on that block.

    python nn/online.py data/enwik7 --side /tmp/e7.side
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import Vocab, LETTER  # noqa: E402


# ---------------- model ----------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn = RMSNorm(d // h)
        self.kn = RMSNorm(d // h)
        self.n2 = RMSNorm(d)
        self.w12 = nn.Linear(d, 2 * ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        q = rope(self.qn(q), cos, sin).transpose(1, 2)
        k = rope(self.kn(k), cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        return x + self.w3(F.silu(g) * u)


class LM(nn.Module):
    def __init__(self, V, d, L, h, ff, ctx):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)
        self.blocks = nn.ModuleList(Block(d, h, ff) for _ in range(L))
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.qhead = nn.Linear(d, 256, bias=False)  # lookahead: byte after the run
        hd = d // h
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(ctx).float()[:, None] * inv[None]
        self.register_buffer("cos", ang.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", ang.sin()[None, :, None, :], persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.o.weight)
            nn.init.zeros_(b.w3.weight)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        return self.head(x), self.qhead(x)


# ---------------- marginalisation ----------------

class Marginaliser:
    """Turns token distributions into per-bit probabilities for byte records."""

    def __init__(self, vocab: Vocab, dev):
        self.dev = dev
        self.lo = torch.from_numpy(vocab.lo).to(dev)
        self.hi = torch.from_numpy(vocab.hi).to(dev)
        self.exact = torch.from_numpy(vocab.exact).to(dev)
        self.sorted_ids = torch.from_numpy(vocab.sorted_ids).to(dev)
        self.byte_sorted = torch.from_numpy(vocab.byte_sorted).to(dev)
        letter = torch.from_numpy(LETTER.copy()).to(dev)
        self.letter = letter
        self.nonletter = ~letter
        # generic OOV continuation: letters uniform (0.7), delimiter via Q (0.3)
        self.u_letter = letter.float() * (0.7 / letter.sum())

    @torch.no_grad()
    def bits(self, logits, qlogits, rtok, rnode, rkind, rfirst, rbyte):
        """logits [M, V] (rows = token positions referenced by records),
        rtok indexes those rows. Returns uint16 [R, 8]."""
        P = torch.softmax(logits.float(), -1)
        Ps = P[:, self.sorted_ids]
        cum = torch.zeros(Ps.shape[0], Ps.shape[1] + 1, device=self.dev, dtype=torch.float64)
        cum[:, 1:] = torch.cumsum(Ps.double(), -1)
        Q = torch.softmax(qlogits.float(), -1)
        Qn = Q * self.nonletter
        Qn = Qn / Qn.sum(-1, keepdim=True).clamp_min(1e-30)

        W = cum.shape[1]
        cf = cum.reshape(-1)
        base = (rtok.long() * W)[:, None]

        def mass(lo_idx, hi_idx):
            return (cf[base + hi_idx] - cf[base + lo_idx]).float()

        lo = self.lo[rnode].long()
        hi = self.hi[rnode].long()
        D = mass(lo, hi).clamp_min(0)  # child masses [R,256]
        k1 = rkind == 1
        if k1.any():
            ex = self.exact[rnode]
            exc = ex.clamp_min(0)[:, None]
            exm = torch.where(ex >= 0, mass(exc, exc + 1)[:, 0], torch.zeros_like(ex, dtype=torch.float32))
            fb = self.byte_sorted[rfirst.long()][:, None]
            oov = mass(fb, fb + 1)[:, 0]
            Qr = Qn[rtok]
            add = exm[:, None] * Qr + oov[:, None] * (self.u_letter[None] + 0.3 * Qr)
            D1 = D * self.letter + add
            D = torch.where(k1[:, None], D1, D)
        D = D + 1e-6
        C = torch.zeros(D.shape[0], 257, device=self.dev, dtype=torch.float64)
        C[:, 1:] = torch.cumsum(D.double(), -1)
        y = rbyte.long()
        out = torch.empty(D.shape[0], 8, device=self.dev, dtype=torch.int32)
        for k in range(8):
            pre = y >> (8 - k)
            w = 1 << (8 - k)
            a = pre * w
            m = a + w // 2
            b = a + w
            tot = C.gather(1, b[:, None]) - C.gather(1, a[:, None])
            one = C.gather(1, b[:, None]) - C.gather(1, m[:, None])
            p = (one / tot.clamp_min(1e-300))[:, 0]
            out[:, k] = (p * 65536).round().clamp(1, 65535).int()
        return out.to(torch.int32)


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--side", default=None)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--qw", type=float, default=0.3, help="lookahead head loss weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    data = open(args.file, "rb").read()
    n = len(data)
    t0 = time.time()
    vocab = Vocab.build(data, args.vocab)
    toks, qtgt, rtok, rnode, rkind, rfirst = vocab.tokenize(data)
    assert vocab.detokenize(toks.tolist()) == data, "tokenizer not reversible"
    ntok = len(toks)
    print(f"bytes {n}  tokens {ntok}  ({n/ntok:.2f} B/tok)  vocab {vocab.V}  nodes {vocab.n_nodes}  "
          f"records {(rkind>=0).sum()}/{n}  prep {time.time()-t0:.1f}s", flush=True)

    V, PAD, T, B = vocab.V, vocab.PAD, args.T, args.B
    model = LM(V, args.d, args.layers, args.heads, args.ff, 2 * T).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0, eps=1e-8)
    marg = Marginaliser(vocab, dev)

    # streams: B contiguous token chunks
    per = math.ceil(ntok / B)
    steps = math.ceil(per / T)
    padded = np.full(B * per, PAD, dtype=np.int64)
    padded[:ntok] = toks
    S = torch.from_numpy(padded.reshape(B, per)).to(dev)
    qpad = np.full(B * per, -1, dtype=np.int64)
    qpad[:ntok] = qtgt
    Qt = torch.from_numpy(qpad.reshape(B, per)).to(dev)
    left = torch.full((B, T + 1), PAD, dtype=torch.long, device=dev)
    Sx = torch.cat([left, S, torch.full((B, steps * T - per + 1), PAD, dtype=torch.long, device=dev)], 1)
    Qx = torch.cat([torch.full((B, T + 1), -1, dtype=torch.long, device=dev), Qt,
                    torch.full((B, steps * T - per + 1), -1, dtype=torch.long, device=dev)], 1)

    # byte records grouped by the token they reference
    has = np.flatnonzero(rkind >= 0)
    rec_order = has[np.argsort(rtok[has], kind="stable")]
    rec_t = rtok[rec_order]
    rec_bounds = np.searchsorted(rec_t, np.arange(0, ntok + 1))
    side = np.zeros((n, 8), dtype=np.uint16) if args.side else None
    data_np = np.frombuffer(data, dtype=np.uint8)

    print(f"params {nparam/1e6:.2f}M  streams {B}x{per}  steps {steps}", flush=True)
    total_bits = 0.0
    sched_warm = 50
    for step in range(steps):
        a = step * T  # window covers stream positions [a-T, a+T) ; predict [a, a+T)
        inp = Sx[:, a: a + 2 * T]          # tokens at stream pos a-T-1 .. a+T-2
        tgt = Sx[:, a + 1: a + 2 * T + 1]  # tokens at a-T .. a+T-1
        qt = Qx[:, a + 1: a + 2 * T + 1]
        lr = args.lr * min(1.0, (step + 1) / sched_warm)
        for g in opt.param_groups:
            g["lr"] = lr
        for it in range(args.inner):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, ql = model(inp)
            lg = logits[:, T:].float()
            y = tgt[:, T:]
            valid = y != PAD
            loss_tok = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1).clamp_max(V - 1), reduction="none").view(B, T)
            if it == 0:
                with torch.no_grad():
                    total_bits += (loss_tok * valid).sum().item() / math.log(2)
                    if side is not None:
                        # map (stream b, offset j) -> global token index
                        gpos = torch.arange(B, device=dev)[:, None] * per + (a + torch.arange(T, device=dev))[None]
                        gpos_np = gpos.cpu().numpy().reshape(-1)
                        okm = (gpos_np < ntok) & valid.cpu().numpy().reshape(-1)
                        rows, rt_local = [], []
                        for r, g in enumerate(gpos_np):
                            if okm[r]:
                                s0, s1 = rec_bounds[g], rec_bounds[g + 1]
                                if s1 > s0:
                                    rows.append(rec_order[s0:s1])
                                    rt_local.append(np.full(s1 - s0, r))
                        if rows:
                            rb = np.concatenate(rows)
                            rr = torch.from_numpy(np.concatenate(rt_local)).to(dev)
                            flat_l = lg.reshape(B * T, V)
                            flat_q = ql[:, T:].reshape(B * T, 256)
                            out = marg.bits(flat_l, flat_q, rr,
                                            torch.from_numpy(rnode[rb]).to(dev),
                                            torch.from_numpy(rkind[rb]).to(dev),
                                            torch.from_numpy(rfirst[rb]).to(dev),
                                            torch.from_numpy(data_np[rb].copy()).to(dev))
                            side[rb] = out.cpu().numpy().astype(np.uint16)
            qy = qt[:, T:]
            qmask = qy >= 0
            if qmask.any():
                lq = F.cross_entropy(ql[:, T:].float()[qmask], qy[qmask])
            else:
                lq = torch.zeros((), device=dev)
            loss = (loss_tok * valid).sum() / valid.sum().clamp_min(1) + args.qw * lq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % 50 == 0 or step == steps - 1:
            done = min(ntok, (step + 1) * T * B)
            el = time.time() - t0
            print(f"  step {step}/{steps}  ~{done/ntok*100:5.1f}%  NN {total_bits/8/1e6:.3f} MB so far  "
                  f"loss {loss.item():.3f}  {el:.0f}s", flush=True)

    secs = time.time() - t0
    nn_bytes = total_bits / 8
    print(f"NN-only: {nn_bytes:.0f} bytes  ({total_bits/n:.4f} bpc)  time {secs:.0f}s  "
          f"peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if side is not None:
        side.tofile(args.side)
        print(f"side file: {args.side}")
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps({"ts": int(time.time()), "kind": "nn", "file": args.file, "in": n,
                                "out": round(nn_bytes), "bpc": total_bits / n, "secs": round(secs, 1),
                                "note": args.note, "rev": open(".rev").read().strip() if os.path.exists(".rev") else "",
                                "cfg": {k: v for k, v in vars(args).items() if k not in ("file", "log", "side")},
                                "params": nparam}) + "\n")


if __name__ == "__main__":
    main()
