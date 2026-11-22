# xzip — Plan

**Goal:** a lossless compressor that sets a new best total size on `enwik9` under Large Text Compression Benchmark (LTCB) rules, and runs on **this machine**.
**Written:** 2026-11-22. Revise it at every phase gate.

---

## 0. Ground truth (checked 2026-11-22)

### The leaderboard we are entering

From the full LTCB table (pasted by the user on 2026-11-22) and tufalabs.ai/research/tufazip. **ex-girlfriend 2.0 is #1.**

| LTCB rank | Entry | enwik9 total (bytes) | Comp / Decomp (≈ s on enwik9) | RAM | Notes |
|---|---|---|---|---|---|
| **1** | **ex-girlfriend 2.0** | **88,983,124** | 340k / 342k (~4 days) | 24 GB | transformer + CM hybrid. **The target.** |
| 2 | cmix-neif 2.0 | 89,346,392 | 177k / 177k (~2 days) | 9.8 GB | Transformer + online LSTM + statistical models |
| 4 | cmix-wolf-p14 | 90,735,088 | 125k (~1.5 days) | 9.8 GB | |
| 10 | fx2-cmix-t-ft8 | 96,945,005 | 112k / 130k | 9.7 GB | |
| 11 | fx2-cmix-transformer | 96,996,198 | 134k | 9.5 GB | 6M-param offline transformer at 4-bit, plus PPMd and CM |
| 13 | **tufazip v2** | 105,388,353 | 648k / 630k (~7.5 days) | 74 GB | pure neural, 163M params, 192-token context, 35M-token replay, 4.43 MB prior + decoder |
| 14 | DNA-4 | 105,554,360 | 71k / 9.5k | 21 GB | asymmetric (decompression ~2.6 h) |
| 15 | altxs 1.0.0 | 106,924,811 | 227k / 1.19M | 88 GB | **93.43 MB payload** but a 13.49 MB decompressor. Big priors cut the payload a lot, then the weight cost wipes the gain out. |
| 16 | nncp v3.2 | 107,261,318 | 242k | 7.6 GB | pure neural |
| 19 | cmix v21 | 108,244,767 | 623k | 31 GB | classic CM |
| 26 | starlit | 114,951,433 | 174k | 10 GB | CM plus article reordering (2021) |

**What this tells us**

- tufazip is the best *pure-neural* entry, but it sits about **16 MB behind** the hybrids. The frontier is hybrid: transformer plus context mixing (CM) plus match models. xzip is a hybrid from day one.
- The top entries take **1.5–4 days** per run and use **~10 GB** of RAM. That is our class.
- Compressed weights and dictionaries count toward the total. A source-code decompressor ("s") is counted by its source size.

### Our hardware (the binding constraint)

| | |
|---|---|
| GPU | RTX 3060 Ti, **8 GB**, Ampere (bf16/fp16 tensor cores) |
| CPU | Ryzen 5 3600X, 6 cores / 12 threads |
| RAM | **16 GB** (about 12 GB usable) |
| Toolchain | Python 3.13, Rust 1.98, CUDA 13.2 driver, Windows 10 |

**Budget we design to:** one full enwik9 run in **≤ 4 days**, in each direction. That gives:

- **GPU:** about 1–2×10¹⁸ FLOPs, roughly 1/50 to 1/100 of tufazip's ~10²⁰. We cannot win by brute force, so every idea has to buy bits per FLOP.
- **CPU:** a single-threaded CM gets about 300 µs per byte. cmix takes about 600 µs per byte, and paq8px about 50 µs.
- **Memory:** CM tables plus match index plus history ≤ ~10 GB of RAM. Model, optimizer state and KV cache ≤ ~7 GB of VRAM.
- **Iterations:** about 6 full enwik9 runs per month. So almost all research happens on smaller files, and only promoted ideas ever run on enwik9.

---

## 1. Thesis: where xzip finds its bytes

tufazip's own results show where its bytes go.

1. **Replay works, but uniform replay is wasteful.** Going from 15M to 35M tokens of replay saved 1.30 MB for 40% more FLOPs. Most replayed tokens are already predicted well.
2. **Neural nets are poor at verbatim recall.** The ~16 MB gap to the hybrids is mostly exact-repeat structure: templates, names, links, XML headers. A match model or CM gets that for almost nothing.
3. **Prior weights pay off only when they are tiny.** L6 predicted better than C2 but lost on total size because of its weight bytes. The value is in *compact* priors.
4. **File order is arbitrary.** Reordering articles by similarity is already proven: the STARLIT Hutter Prize entry did it. It moves information closer to where it is used.

So xzip is **an online-learned transformer (GPU) fused at bit level with a strong CM and match-model stack (CPU), fed by reversible enwik-specific preprocessing.** Our original contributions are the six bets in §3.

---

## 2. Architecture

```
                ┌──────────────── ENCODER ONLY ────────────────┐
enwik9 ──► [P1] reversible preprocessing ──► [P2] article reorder ──► token/byte stream
                (XML/entity/markup normalise,      (embedding sim + greedy TSP,
                 capitalisation flags, dict)        permutation stored in header)
                └──────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────────┐
          ▼                        ▼                              ▼
 [M1] GPU Transformer      [M2] CPU CM stack                [M3] Match / long-range
  online-learned, B lock-   order-N, word, sparse, XML/       hash + suffix match over ALL
  step streams, optional     wiki models, indirect, field     decoded history (all streams),
  compact prior             models (ids/timestamps)           several minimum lengths
          │ token dist          │ bit preds                       │ bit preds
          ▼                     │                                 │
 [F1] trie marginalisation ─────┴──────────► [MIX] gated mixer network
  token dist → P(next bit | byte prefix)     (mixer context includes transformer hidden state)
                                                   │
                                              [SSE/APM chain]
                                                   │
                                          binary arithmetic coder ──► .xz9
```

**Key design decisions**

- **Fusion happens at the bit level, not the token level.** The transformer predicts word tokens (tufazip found a 4K vocabulary best). We turn its token distribution into next-bit probabilities by summing over a trie of the vocabulary, given the bytes of the current token decoded so far. That makes the transformer just one more strong input to a paq/cmix-style mixer, so CM, match and transformer predictions mix bit by bit.
- **Lockstep multi-stream learning.** The (reordered) file is split into B contiguous streams, with B around 16–64. They are processed in lockstep so the GPU always has a batch. The transformer learns from all streams with shared weights, as in NNCP. The **CM and match model run single-threaded over the interleaved byte order, with shared tables.** Each stream keeps its own order-N context hashes, but every stream sees every other stream's history. That keeps the CM deterministic without locks, and gives us cross-stream long-range memory.
- **Online learning in macro-blocks, as in NNCP.** Within a block, weights are frozen. During decoding we predict with an incremental KV cache. At the block boundary we take one deterministic optimizer step, plus any replay steps.
- **Determinism**
  - The CM is all integer and fixed-point, as in paq.
  - The GPU side uses fixed shapes, no atomics, `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`, on the same GPU model for encoding and decoding. LTCB allows this.
  - Mixer inputs that come from the GPU are **quantized to 12-bit fixed point before they cross to the CPU**, so floating-point noise can't leak into the coder.
- **Stack**
  - **Rust** (`xzip-core`) for the arithmetic coder, CM, match model, mixer, preprocessing and the bit-exact codec. It is exposed to Python through PyO3.
  - **Python + PyTorch** (`xzip-nn`) for the transformer.
  - The submission is source plus compressed weights. PyTorch is a public dependency and is not shipped.
  - Native Windows first. Move to WSL2 only if we need Triton or `torch.compile`.

---

## 3. The six xzip bets (ranked by expected bytes per unit of effort)

Each bet has a hypothesis, a cheap test, and a **kill criterion**. The test comes before any effort to make it fast.

| # | Bet | Hypothesis | First test | Kill if |
|---|---|---|---|---|
| **B1** | **Neural mixer context** | Using the transformer's hidden state (projected to about 16 dims) to select or gate mixer weights beats feeding it in only as a prediction. | enwik7, CM+NN, with vs. without hidden-state gating | < 0.5% gain |
| **B2** | **Surprise-prioritized replay** | Replaying only segments with high recent loss (prioritized, with a decaying priority score) matches uniform 35M-token replay at 3–5× fewer FLOPs. | enwik8, transformer alone, same FLOP budget: uniform vs. prioritized | ≤ uniform at equal FLOPs |
| **B3** | **Semantic article reordering** | Ordering articles by embedding similarity (greedy nearest neighbour, then 2-opt) beats file order and STARLIT-style ordering, even after paying for the permutation (about 243k articles, roughly 0.3–0.6 MB before compressing it). | enwik8 reordered vs. original, full pipeline, permutation cost included | net < 0.3% |
| **B4** | **Cross-stream match memory** | One match index over all decoded streams makes up for the context lost by splitting the file into B streams, so large B (fast) costs almost nothing. | enwik8: B = 1 / 16 / 64, with the match model per-stream vs. global | the B = 64 penalty stays > 1% |
| **B5** | **Compact procedural prior** | A prior built from shared blocks plus small per-layer deltas (low-rank or ternary), stored at ≤ 1.5 MB, keeps most of the L6 gain. That is tufazip's open question. | enwik9 *prefix* (first 200 MB): no prior vs. C2-like vs. our prior, weight bytes counted | net gain < 0.3 MB on enwik9 |
| **B7** | **Encoder-steered online learning** ([ALGORITHMS.md](ALGORITHMS.md) §E1) | The encoder looks ahead, picks the best of k training actions per block (replay or not, LR scale) and codes the choice. That costs about 10–20 KB of side information and changes nothing on the decoder's side. It beats any fixed replay schedule. | enwik8: k = 2 ("replay step yes/no") vs. always/never, side-information bits included | net gain < 0.3% |
| **B6** | **Structured field models** | Dedicated predictors for the XML revision headers (`<id>` increments, `<timestamp>` deltas, contributor IDs, recurring infobox and template schemas) beat generic context models. This is our practical "program synthesis" bet. | enwik8 bits spent in header regions, before and after | < 0.2% |

**Watch-list (not scheduled yet):** test-time-training or fast-weight layers (TTT-Linear, Gated DeltaNet) as a cheaper stand-in for part of replay; kNN-LM over hidden states (memory is the problem: 300M keys); hypernetwork-generated layers; BLT-style byte patching (it matters for universality, not for enwik9).

**Deliberately deprioritized for the enwik9 track:** asymmetric fast decoding and universality beyond text. LTCB ranks on size and both kinds of top entry are symmetric. These go to the "practical track" (§7) after we have a result.

---

## 4. Research method (how we avoid fooling ourselves)

- **Measure ideal code length, not archives.** During research, score = Σ −log₂ p(bit), counted in bytes. The arithmetic coder adds only a few bytes on top. A real encode/decode roundtrip runs only at a phase gate. This makes iteration 2–3× faster and separates bugs from ideas.
- **A ladder of file sizes.** enwik8 is exactly the first 100 MB of enwik9, so every rung is a prefix of the final target.

  | Rung | Size | Wall-clock target | Use |
  |---|---|---|---|
  | `e6` | 1 MB | minutes | unit or regression checks |
  | `e7` | 10 MB | ≤ 1 h | screening ideas |
  | `e8` | 100 MB | overnight | confirming ideas (B2–B4) |
  | `e9p` | first 200–300 MB of enwik9 | 1–1.5 days | priors and scaling (B5) |
  | `e9` | 1 GB | ≤ 4 days | phase gates only |

  tufazip found that conclusions about priors **flip between enwik8 and enwik9**. So anything that depends on scale (priors, model size, replay size) has to be confirmed on `e9p` or larger.
- **Report bits per region.** Every run reports bytes per region type: XML header, article text, links, templates and tables, and other markup. It also reports how much each mixer input contributes (ablation by zeroing each input's weight). Then we know *where* an idea wins.
- **Equal-budget comparisons only.** Every A/B states its FLOPs, wall-clock time and RAM, and compares at equal budget.
- **Keep a results log.** Store one row per run in `results/runs.jsonl`: config hash, git SHA, rung, bytes, region breakdown, time and memory. No result counts unless it is in the log.

---

## 5. Phases and gates

Gate targets are in **enwik9 total bytes**. Before a full run they are projected from the scaling across `e8` and `e9p`, and confirmed at each gate by an actual run.

### Phase 0: Foundations (week 1)
- Run `git init` and create the repo layout (§6). Download enwik8 and enwik9 and check their hashes.
- Measure baselines on our machine: `xz`, `zstd --ultra -22`, and **cmix v21 and fx2-cmix on `e7` and `e8`**. These calibrate how fast this CPU is against the LTCB timings, and give us a strong reference.
- Find out which top-10 entries have public source and how they work, ex-girlfriend and cmix-neif first. Pick a **reference implementation** to compare against and learn from. Licensing note: cmix and paq are GPL. Porting their ideas is fine, and any copied code makes xzip GPL.
- **Gate:** the harness reports ideal code length plus a region breakdown on `e6` and `e7`, and the baselines are logged.

### Phase 1: CM core in Rust (weeks 2–4)
- A binary arithmetic coder (32-bit, carryless) with **≥ 16-bit (ideally 24-bit) probabilities**, and a real encode/decode roundtrip test from the start. At 12-bit precision the clamping alone costs up to about 350 KB on enwik9 (see [ALGORITHMS.md](ALGORITHMS.md) §A3).
- The models:
  - order-1 to order-N hashed contexts with state machines
  - word and sparse models
  - match model (hash plus suffix verification, several minimum lengths)
  - XML/wiki context models
  - two-layer gated mixer
  - SSE/APM chain
- Preprocessing P1: XML entity decoding, capitalization flags, and a word dictionary transform. **Each transform is tested for exact roundtrip on all of enwik9.**
- **Gate:** `e8` ≤ about 17.5 MB (paq8px class) at ≤ 300 µs/byte, and a bit-exact roundtrip on `e8`.

### Phase 2: Neural model plus fusion (weeks 4–8)
- `xzip-nn`: a decoder-only transformer with RoPE, SwiGLU, RMSNorm and QK-norm (tufazip's recipe).
  - Start at about 40–80M params, 4K word vocabulary, 256–512 token context.
  - Training is online in macro-blocks with B lockstep streams.
- F1: trie marginalization from token distributions to next-bit predictions. Mixer inputs are quantized to fixed point.
- Deterministic GPU decoding: test a full `e7` encode → decode → byte compare.
- Test the optimizers first: AdamW vs. **Muon** vs. **schedule-free AdamW**, plus decay toward the prior. Learning faster per token directly means fewer bytes (see [ALGORITHMS.md](ALGORITHMS.md) §A2 and §C1–C2).
- Test **B1** (via GLN halfspace gating, §D3) and **B4** here.
- **Gate:** projected enwik9 total **< 105.39 MB**, beating tufazip, within the 4-day budget. Confirmed by a real `e9` run.

### Phase 3: Long-range bets (weeks 8–12)
- **B2** surprise-prioritized replay (scored by reducible loss), **B7** encoder-steered training, **B3** semantic reordering (encoder side only), **B6** field models.
- Tune model size, B and context length against wall-clock time. The per-token budget is set by the CM thread (~300 µs/byte), so the GPU is sized to keep pace with it, no more.
- **Gate:** projected total **< 95.79 MB** (top 6, beating makimono). Confirmed by a real `e9` run.

### Phase 4: Priors and polish (weeks 12–16+)
- **B5** compact prior. Train it offline on enwik9 itself (allowed; we pay for its bytes). Try quantization-aware training at 1.58–4 bits and entropy-code the weights.
- Tuning: learning-rate and optimizer schedule, the mixer learning rate, the SSE chain, and a dictionary chosen to suit the transformer vocabulary.
- **Gate:** total **below the current #1**, using RAM under the 16 GB machine limit. Then do the full roundtrip on enwik9 and package the submission.

**If a gate fails:** stop. Do a region-breakdown post-mortem against the reference implementation and re-plan. Don't pile more bets on top.

---

## 6. Repo layout

```
xzip/
  PLAN.md
  crates/
    xzip-core/      # Rust: coder, CM models, match, mixer, SSE, preprocessing, PyO3 bindings
    xzip-cli/       # Rust: `xzip c|d|eval` entry point
  nn/               # Python: transformer, online trainer, replay, trie marginalisation
  harness/          # Python: rung runner, ideal-codelength eval, region breakdown, results log
  data/             # enwik8, enwik9 (git-ignored) + sha256 manifest
  results/runs.jsonl
  baselines/        # scripts to build/run cmix, fx2-cmix, xz, zstd
```

---

## 7. Practical track (after the enwik9 result)

Once the enwik9 entry exists, reuse its parts for a general-purpose tool:

- an asymmetric mode (a heavy encoder that chooses the transforms and order, plus a lighter decoder)
- BLT-style byte patching so it handles arbitrary file types
- a reversible transform search for each file type (the "autonomous compression scientist" idea)

This track is out of scope until Phase 4's gate.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| GPU nondeterminism breaks decoding | Fixed shapes, deterministic flags, fixed-point mixer inputs. A roundtrip test at every gate, starting in Phase 2. |
| 16 GB RAM caps the CM, match index and history | Tables sized to budget, a compact suffix structure, history stored once. Measure peak RSS on every run. |
| 4-day runs make iteration too slow | The rung ladder, plus projecting from `e8` and `e9p`. Full `e9` runs only at gates. Option: rent a larger GPU for the final runs only. |
| Top entries move (89 MB entries appeared in Nov 2026) | Check the leaderboard at each gate. Targets are relative to the current #1. |
| Ideas already used by top entries (B1, B3 likely in part) | Phase 0 study of the reference implementation. Our edge is the combination and B2, B4 and B5. |

---

## 9. Status log

**Test box** (from 2026-11-22): Fedora 44 laptop, Ryzen 7 250 (8 cores / 16 threads), 30 GB RAM, RTX 5050 Laptop 8 GB. Code is written on the Windows PC and run there via `tools/remote.py`. Data lives in `~/xzip/data` and every run is logged to `results/runs.jsonl`.

| Date | Milestone | enwik7 | enwik8 |
|---|---|---|---|
| 11-22 | xz -9e / zstd -22 --long | 2,720,256 / 2,793,530 | 24,831,648 / 25,273,903 |
| 11-22 | CM v0: orders 0–8, word, match, 2-layer mixer, 2 APMs | 2,265,186 | 20,970,757 |
| 11-22 | CM v1: checksummed nibble buckets, run model, inheritance, sparse + column contexts | 2,079,922 | |
| 11-22 | CM v2: indirect bit-history, 2 match models, word trigram | 2,038,239 | |
| 11-22 | CM v3: bracket, line-head and word-skip contexts, confidence inputs, char-class mixer, tuned learning rate | 1,993,246 | **18,160,029** (10.5 µs/B, 3.9 GB) |

Phase 2 groundwork is in place:
- `nn/tokenizer.py`: reversible word tokenizer plus vocabulary trie.
- `nn/online.py`: online transformer with lockstep streams and GPU token→bit conversion (§B7, using a lookahead head for the byte after each word).
- `xzip eval --side`: mixes the neural predictions into the CM.
