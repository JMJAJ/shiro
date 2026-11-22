# xzip

A research log for one question: can a small online transformer and a classic context-mixing compressor, run together, set a new best total on enwik9 under the rules of the [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html)?

The bar is high. The top entry today is **ex-girlfriend 2.0 at 88,983,124 bytes**. The pure-neural entry that started this project, tufazip, sits at 105.39 MB. Everything below was measured on one Fedora laptop: Ryzen 7 250, 30 GB RAM, and an RTX 5050 Laptop GPU with 8 GB.

This page is the story so far: what was tried, what moved the number, what didn't, and our best guess at why. The long-form plan is in [PLAN.md](PLAN.md) and the math in [ALGORITHMS.md](ALGORITHMS.md). Every run ever made is one line in [results/runs.jsonl](results/runs.jsonl).

---

## Where it stands

| | enwik7 (10 MB) | enwik8 (100 MB) |
|---|---|---|
| xz -9e | 2,720,256 | 24,831,648 |
| CM v3 (end of day 1) | 1,993,246 | 18,160,029 |
| Transformer only (online, 23M params) | 2,526,495 | 19,438,897 |
| **CM v3 + transformer, fused** | **1,938,957** | **16,841,487** |
| CM v4.2 (today, CM only) | 1,961,099 | *running* |
| CM v4.2 + transformer, 2 training passes per batch | | *running* |

**Read these numbers with care.** The sizes are the model's code length on the file, Σ −log₂ p over every bit. They are not yet the size of an archive that decompresses and is counted with its decompressor. The CM alone already round-trips through a real arithmetic coder with bit-exact deterministic math. The hybrid does not yet: a decoder would have to replay the GPU training exactly, and getting that deterministic is still to do. Until then, hybrid numbers are research numbers.

---

## The idea

Context mixing (CM, the paq/cmix family) is very good at local, exact statistics: repeated strings, markup, numbers, the structure of wiki templates. It learns from a single example. Transformers are good at what CM is bad at: semantics, long-range topic and grammar. They need many examples to get there.

So xzip runs both. The transformer is trained *online*, predicting each token before it learns from it. Its prediction is converted into one probability per bit and fed to the CM's mixer as one more input. The mixer learns, context by context, when to trust which model. The bet is that the two models make different mistakes, so the mix beats either one alone by more than the cost of running both.

That bet has held so far. On enwik8 the transformer alone is *worse* than the CM (19.44 MB vs 18.16 MB), yet adding it cuts the CM's output by **7.3%**.

---

## Journal

### Day 1: building the CM from scratch

The CM is written in Rust: [crates/xzip-core](crates/xzip-core). The fundamentals came first:

- a 32-bit carryless binary arithmetic coder with 24-bit probabilities;
- `exp` and `ln` built from basic IEEE operations only ([dmath.rs](crates/xzip-core/src/dmath.rs)), so the output is bit-identical on Windows and Linux. A compressor whose decoder disagrees with its encoder in the last float bit is useless, so this went in before any modelling.

Then the model, in four rounds. All numbers are enwik7.

| Version | What changed | enwik7 | Δ |
|---|---|---|---|
| v0 | orders 0–8, a word context, one match model, 2-layer mixer, 2 APMs | 2,265,186 | |
| v1 | 64-byte checksummed nibble buckets, a run model, PPM-style inheritance, sparse and column contexts | 2,079,922 | −8.2% |
| v2 | indirect bit-history maps, a second match model (min length 16), word trigram | 2,038,239 | −2.0% |
| v3 | bracket, line-head and word-skip contexts, confidence inputs, char-class mixer, tuned learning rates | 1,993,246 | −2.2% |

**What landed and why.**

- **v1 gave the biggest jump.** The table layout was most of it. One 64-byte cache line per (context, nibble) holds all 15 bit-slots of the nibble's binary tree, so a lookup costs one cache miss instead of 15. A checksum lets a colliding context be detected and replaced rather than silently polluting statistics.
- **Inheritance** helps every time. A new order-n context starts from its order-(n−1) parent's probabilities instead of 0.5. That matters on wiki text, where most high-order contexts are seen only a handful of times.
- **The run model** remembers the last byte seen in each context and how many times in a row it repeated. It is cheap, and it covers the deterministic contexts that adaptive counters learn slowly.

**What didn't.** Parameter sweeps (learning rates, count limits, mixer init, APM rates) gave about 0.1% each and quickly flattened out. Every real gain on day 1 came from structure, not knobs. That pattern has held ever since.

### Day 1: the neural side

[nn/tokenizer.py](nn/tokenizer.py) and [nn/online.py](nn/online.py).

- **Tokenizer.** Maximal ASCII letter runs that appear in the vocabulary become one token. Everything else, including out-of-vocabulary letters, is one byte per token. The vocabulary has 16,384 entries and is ranked by count × (length − 1). It is fully reversible and averages **1.90 bytes per token** on enwik8.
- **Model.** A small modern transformer: RoPE, RMSNorm, QK-norm, SwiGLU, zero-initialised output projections. d = 384, 6 layers, 23M parameters, AdamW at lr 1e-3.
- **Streams.** The file is split into B contiguous streams trained in lockstep, NNCP-style, so the GPU has a batch. Each window predicts T = 256 new tokens with T tokens of context.

**The hard part was token → bit.** The CM wants P(next bit). The transformer gives P(next token). The bridge is a trie over the vocabulary. At each byte, the probability mass of all vocabulary strings that continue the current prefix with that byte gives a 256-way byte distribution, and that becomes 8 bit probabilities.

Two details took real thought:

1. **A decoder can't consume a word token until it has decoded the delimiter after it.** It doesn't know the word ended. So every byte inside a word, and the delimiter right after it, must be predicted from the distribution of the word's *first* token. A small second head on the model (the lookahead head) predicts the byte that follows the word. It handles the delimiter.
2. **Out-of-vocabulary words.** Once the prefix leaves the trie, the transformer has nothing useful to say, so those bytes get no prediction and the CM is on its own. That covers about 5% of the bytes in enwik8.

**What failed on the way.** The GPU ran out of memory twice at B = 32:

- First in the marginaliser. Fixed by processing token rows in chunks of 1024.
- Then in the backward pass. Fixed by applying the output heads only to the half of the window being predicted, and using expandable CUDA segments.

Both fixes were checked to give byte-identical results on a 200 KB test file.

**The day-1 result.** The fusion worked on the first try: enwik7 dropped from 1,993,246 to 1,938,957 (**−2.7%**). Overnight, enwik8 gave **16,841,487**, which is 7.3% below CM v3 and past the Phase 1 target of 17.5 MB. The gain *grows* with file size, as it should. The transformer is a slow learner and has 10× more data to learn from on enwik8.

### Day 2: more CM structure

Two goals: a stronger CM partner, and faster ways to test fusion.

| Version | What changed | enwik7 | enwik8 |
|---|---|---|---|
| v3 | (baseline) | 1,993,246 | 18,160,029 |
| v4 | orders 12 and 16; indirect order-2 context; final mixer selected by the partial byte; word-length mixer; 12-bit order-2 APM; learned final blend | 1,972,579 | 17,926,130 |
| v4.1 | XML-tag, number, sparse 1+3 and markup-segment contexts; history map indexed by confidence; order-3 APM | 1,963,772 | 17,835,443 |
| v4.2 | a mixer whose weights are selected by the previous two bytes (65,536 weight sets) | 1,961,099 | *running* |

**What landed.**

- **New contexts, one at a time, with an off switch for each.** Each of the four day-2 contexts was measured by turning it off alone (`XZIP_TUNE=xoff=...`). Every one paid for itself, but only by 0.01–0.08% each, 0.32% together.
  - The markup-segment context helped most: which wiki construct we're inside (`[[`, `|`, `=`, `{`, list markers), how far into it, and the current word.
  - The number context came next.
  - The XML-tag context barely registered on enwik7. It stays in because enwik9 has proportionally more header structure.
- **Selecting the final mixer's weights by the partial byte** instead of the bit position: −0.18% from a one-line change. With bit position only, the final mixer has to apply one compromise per bit position across all 256 byte prefixes.
- **The byte-pair mixer** (−0.14%). Brute force, and it works.
- **A learned final blend.** The fixed 0.2/0.27/0.33/0.2 average of the APM outputs is replaced by a small mixer, averaged with the old blend. Using the learned blend *alone* was worse than the fixed weights; averaging the two beat both.

**What didn't.**

- **Inheritance for orders 12 and 16** made no difference. By order 12 the parent's statistics are as sparse as the child's, so there's nothing useful to inherit.
- **Raising the mixer learning rate** from the new defaults made things worse.
- **Speed.** It is going the wrong way: 10.5 µs/byte at v3, 13.1 at v4.1, plus 5.9 GB of RAM. At that speed enwik9 takes about 4 hours of CPU. That's still well inside budget, but it can't keep growing like this.

**A bug that taught something.** APM setup was computing the same 33 squash values a million times per table, which took the test suite from 7 s to 47 s. It now computes one row and copies it.

### Day 2: making the transformer learn more per token

A sweep on enwik7, transformer only:

| Setting | enwik7 | Δ | Time |
|---|---|---|---|
| default (lr 1e-3, B = 16, 1 pass) | 2,525,675 | | 111 s |
| lr 2e-3 | 2,538,677 | +0.5% | |
| lr 3e-3 | 2,554,274 | +1.1% | |
| B = 8 | 2,500,868 | −1.0% | 123 s |
| **2 training passes per batch** | **2,442,000** | **−3.3%** | 227 s |
| 2 passes, lr 2e-3 | 2,450,784 | −3.0% | |

**What this says.** The model is not under-stepped, since a bigger learning rate hurts. It is *under-trained per example*. Training twice on each batch is worth 3.3%, far more than any other knob, at double the GPU time.

Fewer streams (B = 8) helps for a related reason. The same data becomes more sequential steps, and each stream's predictions build on a model that has already seen more of that stream's past.

Both point the same way: a small online model is starved of updates, not of capacity. That is exactly the gap that bets **B2** (surprise-prioritized replay) and **B7** (encoder-steered training) in the plan target, which makes them the obvious next step.

**What went wrong getting it onto enwik8.** The 2-pass run ran out of GPU memory at step 0, twice.

- **First crash.** The first pass's activations were still in memory while the second pass ran. Fixed by freeing them between passes.
- **Second crash.** The quick test after the fix ran without the side-output step, so it missed that this step also needs memory. At B = 32 the single-pass run already peaked at 7.2 GB of 7.57 GB, so a second pass can't fit.
- **The run now going** uses B = 24, about 60 min of GPU.

Lesson logged: smoke tests must use the exact production flags, and job watchers must stop on errors, not only on success. One crash sat unnoticed for close to two hours.

### Fusion: where the next gains probably are

Right now the transformer enters the CM in three ways: as one mixer input, as a present/absent flag, and as the selector for one mixer's weight set. That is the bare minimum. Ideas being tested next, against saved prediction files (`~/xzip-side/`), so the fusion can be iterated in 20 minutes without retraining:

- **`sf`:** the transformer's prediction also goes into the final blend, both raw and through an APM keyed by the partial byte and the previous byte. In the code; not yet measured.
- **B1 in the plan:** use the transformer's *hidden state*, not just its prediction, to select the mixer's weights.
- **The 5% of bytes with no transformer prediction** (out-of-vocabulary word interiors). The trie could fall back to the byte-level tokens instead of giving up.

---

## What we believe so far

1. **Structure beats tuning, on both sides.** Not one parameter sweep has produced more than ~0.2%. Every real step came from a new context, a new way to select weights, or a new way for the two models to talk.
2. **The two models really are complementary.** A transformer that is 7% *worse* than the CM alone still cuts 7.3% off the CM. The CM is not merely being outvoted. It is being corrected where it is weak (text accounts for 77% of the output bytes, at 1.54 bpc).
3. **The transformer is update-starved.** Extra passes and fewer streams help; a bigger learning rate doesn't. Better use of each token (replay, steering) looks like the highest-value work.
4. **The unsolved problem is determinism.** The hybrid numbers only count once a decoder can reproduce the GPU training bit-exactly, or the transformer runs in a deterministic integer or fixed-point form. That has to be solved before any enwik9 claim.

---

## Running it

The CM works on its own anywhere Rust does:

```sh
cargo build --release
./target/release/xzip c  input  output.xzp      # compress
./target/release/xzip d  output.xzp  restored   # decompress
./target/release/xzip eval input --bits 22      # code length + per-region breakdown
XZIP_TUNE="m7=0,xoff=8" ./target/release/xzip eval input   # ablations (eval only)
```

The hybrid needs a CUDA GPU and PyTorch:

```sh
tools/hybrid.sh data/enwik8 22 --B 24 --inner 2      # transformer -> side file -> fused CM eval
tools/sweep.sh  data/enwik7 20 "" "m7=0" "fm=0"      # CM knob sweep, in parallel
tools/nnsweep.sh data/enwik7 "" "--inner 2"          # transformer-only sweep
```

[tools/remote.py](tools/remote.py) syncs the tree to the test laptop and runs jobs there in the background. Connection details come from environment variables and are never written to disk.

### The lab

```sh
XZIP_PASS=... python tools/dash/server.py      # then open http://127.0.0.1:8765
```

A local dashboard with four views. It serves on 127.0.0.1 only.

- **Overview:** best results so far, progress charts for enwik7/8, and where the bytes go by region.
- **Runs:** every run, from the local log and the laptop's logs, with config and region breakdown.
- **Live:** CPU, RAM and GPU on the laptop, running processes, and job logs with progress bars.
- **Formula lab:** write a formula in LaTeX, give it curves as expressions and parameters as sliders, and play. It comes seeded with the math the model actually uses: code length, squash/stretch, counter rates, logistic mixing, fusion cost, the learning-rate schedule, two-part MDL and online learning curves.

Formulas and journal notes are saved to [tools/dash/lab.json](tools/dash/lab.json), so they're versioned with the code.

## Layout

```
crates/xzip-core   CM model, arithmetic coder, deterministic math, region classifier
crates/xzip-cli    the xzip binary: c / d / eval
nn/                tokenizer + trie marginaliser, online transformer
tools/             remote runner, sweeps, hybrid pipeline
results/           runs.jsonl: every measurement, with git revision and settings
PLAN.md            the plan, bets B1–B7, phase gates, status log
ALGORITHMS.md      the math
```
