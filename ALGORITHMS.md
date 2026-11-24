# shiro — Math & Algorithms Reference

This is the reference for the formulas and algorithms shiro uses. Every component in [PLAN.md](PLAN.md) maps to a section here.
Notation: `x_t` is the symbol at time t (a bit, byte or token), `p` is a probability, `σ(z) = 1/(1+e^-z)`, `st(p) = ln(p/(1-p))` is the stretch function and `sq = σ` is its inverse (squash), and `y` is the bit that actually occurred.

**What it covers**

- **A. Objective:** what we minimize (A1–A3)
- **B. The prediction stack:** context mixing, match models, the transformer and how it plugs into the mixer (B1–B9)
- **C. Recent ideas (2023–2026)** worth using (C1–C9)
- **D. Older ideas (1995–2019)** that are mostly forgotten now (D1–D8)
- **E. Our own ideas:** E1 encoder-steered online learning (bet **B7**) and E2 fast/slow logits from one network
- **F. Priority order**

---

## A. The objective (what we actually minimize)

### A1. Total description length (two-part MDL)
```
Total = |decompressor source| + |stored weights / dictionaries| + L(data)
L(data) = Σ_t −log₂ p(x_t | x_<t)          (+ ≤ 2 bits arithmetic-coder overhead)
```
Every design question comes down to three things: does an idea lower `L` by more than it adds to the first two terms, and does it fit the time and RAM budget?

### A2. Online learning is a *prequential* code (Dawid 1984)
When the encoder and decoder train the same model on the same history, the weights are never transmitted. `L(data)` is then simply the model's **cumulative online log-loss**. Two consequences:

- **Sample efficiency is compression.** An optimizer that learns faster per token gives a shorter file, so the optimizer is as important as the architecture (see C1 and C2).
- **A prior pays off only if** `ΔL(prior) > |prior bytes|`. tufazip's L6 prior failed this test (−6.8 MB payload, +10.9 MB weights). Their C2 prior passed it (−4.97 MB payload, +4.38 MB weights).

### A3. Coder precision costs real bytes ⚠
If a binary probability is clamped at `1 − 2^-k`, every near-certain bit still costs `≈ 2^-k / ln 2` bits. enwik9 has about 8×10⁹ binary decisions:

| precision k | cost per "certain" bit | worst case on enwik9 |
|---|---|---|
| 12 bits (paq default) | 3.5×10⁻⁴ | **≈ 350 KB** |
| 16 bits | 2.2×10⁻⁵ | ≈ 22 KB |
| 24 bits (32-bit range coder) | 8.6×10⁻⁸ | ≈ 0.1 KB |

**Rule:** use 32-bit carryless range coding with ≥ 16-bit (ideally 24-bit) probabilities, and do the final mixer and SSE arithmetic at ≥ 16 bits. Near the top of the leaderboard, 350 KB is the gap between neighbouring ranks.

---

## B. The prediction stack

### B1. Binary decomposition
```
p(byte) = Π_{i=0..7} p(bit_i | bit_<i, context)
```
Every model outputs `p(bit = 1)`, and the models are combined bit by bit. This is the PAQ/cmix design, and it is what lets very different models (hashed counters, match models, an LSTM, a transformer) share one mixer.

### B2. Adaptive counters and state machines
- **Adaptive-rate counter:** `p ← p + (y − p) / (n + 1.5)`, then `n ← min(n+1, LIMIT)`. With `n` small the counter learns fast (close to the KT estimator). Capping `n` keeps the counter able to follow changes in the data.
- **KT estimator** (a Bayes-optimal baseline for a memoryless source): `p(1) = (n₁ + ½) / (n + 1)`.
- **Bit-history state machines** (paq's `nex` table) map a short history such as "0,0,1" to a state, and an adaptive table maps each state to a probability. This is the *indirect* model: it learns what tends to follow a given history in this context.

### B3. Logistic mixing (PAQ7+): the core formula
```
p = σ( Σ_i w_i · st(p_i) )
w_i ← w_i + η · (y − p) · st(p_i)
```
This update is *exact* gradient descent on log-loss with respect to `w`. Key refinements:
- **Context-selected weight sets.** A small context (order-0 byte, match state, and so on) chooses which weight vector to use. This is exactly a **gated linear network** (D3).
- **Two layers.** Several mixers, each selected by a different context, feed a final mixer.
- **Learning-rate schedule** `η_t = η₀ / (1 + t/T)^½` plus a floor. Getting this right is worth more than it looks.

### B4. SSE / APM: secondary estimation (calibration)
```
q = st(p) quantised into 33 buckets over [−8, 8];   j = bucket, f = fractional position
p' = (1−f)·A[c][j] + f·A[c][j+1]
A[c][j], A[c][j+1] ← move toward y (rate ~1/64–1/128)
```
This recalibrates `p` given a small context `c`, chained 2 to 4 times with the outputs averaged. It is cheap and reliably worth 1–3%. **Apply it to the transformer's output too.** Neural networks trained online are poorly calibrated.

### B5. Match model (verbatim recall)
- **Index:** hash the last `m` bytes for several values of `m` (for example 6, 12, 20 and 32) and keep a pointer to the most recent occurrence. Alternatively, use a **suffix automaton** over the decoded history: it finds the longest repeated context in amortized O(1) per byte and has no hash collisions.
- **Prediction:** the bit that followed the match. Its strength comes from an adaptive table indexed by `(min(len, 31), expected bit, recent hit/miss history)`. Feed `st(p)` and also `±len` into the mixer.
- **Multi-candidate voting** is easy to overlook: keep the top-k earlier occurrences and let them vote, weighting each by recency and by how many tokens of history it agrees with. That helps a lot on wiki templates.
- This is the main tool for closing the ~16 MB gap between tufazip and the hybrids.

### B6. Transformer (online, as in NNCP and tufazip)
- Decoder-only, with RoPE, SwiGLU, pre- and post-RMSNorm, and QK-norm.
- Softmax over the vocabulary: `p(w | h) = exp(zᵀ_w h) / Σ_v exp(zᵀ_v h)`.
- **Macro-block online learning.** Freeze θ for a block of L tokens and code the block with causal attention (during decoding: L incremental KV-cache steps). Then take one optimizer step on the block's loss, plus replay steps.
- **Lockstep streams.** Split the file into B contiguous streams and run them as one batch. Every stream shares θ.

### B7. Token → bit marginalization (the transformer–CM bridge)
The transformer predicts whole word tokens, but the mixer needs `p(next bit)`. Let the bytes decoded so far inside the current token be `s`, and let `P(w)` be the transformer's distribution:
```
p(next byte = b | s) =  Σ_{w : s·b ⊑ w} P(w)             ← token continues
                      + P(w = s) · Q(b)                   ← token ends; b starts the next token
                      ───────────────────────────────────
                        Σ_{w : s ⊑ w} P(w)
```
- Bit probabilities come from the same sums over subtrees of the vocabulary trie. That is O(depth) per bit, using prefix sums over vocabulary entries sorted in trie order.
- `Q(b)` is the distribution of the next token's first byte. It comes from a cheap **lookahead head**, a second softmax over first bytes computed from the same hidden state, trained jointly.
- If the tokenizer uses "▁word" style leading-space tokens and splits on character class, the "token ends here" term becomes almost deterministic, and Q matters much less.

### B8. Byte-level online LSTM or RNN (what the cmix family uses)
A small LSTM (1–2 layers, about 200 cells) trained by truncated BPTT on bytes. It is a cheap neural input that covers the byte-level details the word transformer misses. cmix-neif's "online LSTM" is this.

### B9. Final probability
```
p_final = SSE-chain( Mixer₂( Mixer₁ₐ(…), Mixer₁ᵦ(…), … ) )   clamped to [2⁻²⁴, 1−2⁻²⁴]
```

---

## C. Recent ideas (2023–2026) that apply directly

| # | Idea | The math | Why it helps here |
|---|---|---|---|
| C1 | **Muon optimizer** (Jordan et al. 2024) | Orthogonalize the momentum with Newton–Schulz: `X ← aX + b(XXᵀ)X + c(XXᵀ)²X`, `(a,b,c) = (3.4445, −4.7750, 2.0315)`, 5 iterations, then `W ← W − η·X`. | It learns noticeably faster per token than AdamW, and by A2 that turns straight into bytes. It is deterministic and cheap. **Highest-value cheap test.** |
| C2 | **Schedule-free AdamW** (Defazio et al. 2024) | `y = (1−β)z + βx`; take the gradient at y; `z ← z − η∇`; `x ← (1−c)x + c·z` with `c = 1/t`. | There is no learning-rate schedule to tune for a stream of unknown length. Predict from both x and y and **mix both**, since they are two views of one model. |
| C3 | **Gated DeltaNet / fast-weight memory** (Yang et al. 2024) | `S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t q_t` | Memory and compute are O(1) per token, and memory reaches beyond the 192–512 token context. A hybrid stack (a few attention layers plus DeltaNet layers) could replace part of the replay. |
| C4 | **Titans / test-time-training memory** (Behrouz et al. 2025; Sun et al. 2024) | `M_t = (1−α_t)M_{t−1} + S_t`, `S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; k_t, v_t)` | A memory that updates by **surprise**, the same principle as B2. It writes long-range facts into memory without full backprop. |
| C5 | **RHO-LOSS / reducible-loss selection** (Mindermann et al. 2022) | `priority(i) = ℓ_current(i) − ℓ_irreducible(i)` | This is the right priority for **bet B2**. Replay what is *learnable*, not what is merely high-loss: random IDs and hashes have high loss but nothing to learn. Estimate the irreducible part as the loss of the best mixer input on that segment, or as how slowly its loss has been falling. |
| C6 | **Prioritized replay** (Schaul et al. 2016) | `P(i) ∝ priority_i^α`, `α ≈ 0.6` | Sampling rule for B2. It stays deterministic if we use a seeded PRNG and integer priorities. |
| C7 | **BitNet b1.58** (Ma et al. 2024) | `W̃ = RoundClip(W/γ, −1, 1)`, `γ = mean|W|`; straight-through estimator during training | A ternary prior costs ≤ 1.58 bits per weight before entropy coding. That is the storage format for bet B5. |
| C8 | **Byte Latent Transformer** (Pagnoni et al. 2024) | Start a new patch when a small byte model's entropy `H(x_t | x_<t)` exceeds a threshold | Mainly for the practical, all-file-types track. For enwik9, word tokens already do most of this. |
| C9 | **"Language Modeling Is Compression"** (Delétang et al. 2023) | `L = Σ −log₂ p` | Confirms that **model size counts** (A1): large pretrained models lose once their weights are paid for. The same lesson as tufazip's L6. |

---

## D. Forgotten gems (older ideas nobody in the top 20 obviously uses fully)

| # | Gem | The math | How shiro uses it |
|---|---|---|---|
| D1 | **Dynamic evaluation with decay to the prior** (Krause et al. 2018–19) | `θ ← θ − η·∇ℓ / √(MS + ε) + λ(θ₀ − θ)` | **Pull weights back toward the stored prior** θ₀. Fast adaptation to the current article stops slowly overwriting what the prior knows. It pairs with B5, and is one line of code. |
| D2 | **Fixed-share / switching experts** (Herbster & Warmuth 1998) | After the Bayes update, `w_i ← (1−α)·w_i + α/n` | A top-level mixture that recovers quickly when the best model changes, for example at article boundaries (prose → table → reference list). Use it as a final stage alongside the logistic mixer. |
| D3 | **Gated Linear Networks** (Veness et al. 2017/2021) | Each neuron's weight vector is selected by **halfspace gating** on side information `z`: `c = [sign(h_jᵀ z − b_j)]_j` | A sound way to do **bet B1**. Let `z` be a 16-dim random projection of the transformer's hidden state, and let the halfspaces choose the mixer weight sets. Each neuron is still a convex, local, exact-gradient problem with no backprop through the mixer. |
| D4 | **Partition Tree Weighting** (Veness et al. 2013) | A Bayesian mixture over every binary partition of time into stationary segments. Cost O(log n) per step. | A sound way to forget at regime changes. A cheaper alternative: run counters at two different rates and let the mixer choose. |
| D5 | **Context Tree Weighting** (Willems et al. 1995) | `P_w(s) = ½·P_e(s) + ½·P_w(0s)·P_w(1s)` | Weak on text on its own (211 MB on enwik9). The *idea*, Bayesian averaging over context depths, shows up in D3 and in PPM inheritance. Use it for **binary side streams** such as capitalization flags and the permutation. |
| D6 | **PPM information inheritance** (Shkarin, PPMd) | Initialize a new high-order context's statistics from its parent order, instead of starting from scratch | New hashed context slots start from the lower-order prediction instead of p = ½. That saves the cold-start cost on all of the roughly 10⁸ new contexts. |
| D7 | **Dictionary preprocessing with capitalization flags** (XWRT, drt, phda) | Map frequent words to 1–3 byte codes; replace "Word" with flag + "word" | Proven to be worth several MB on enwik. It also reduces the number of word types the transformer and CM must learn. |
| D8 | **Article reordering** (STARLIT 2021) | Store the permutation π. The raw cost is `log₂ n! ≈ n(log₂ n − 1.443)` bits, which is **about 500 KB for n ≈ 243k articles** before any modelling. | The gain has to clear about 500 KB, minus whatever we save by coding π as deltas between neighbouring original positions with an adaptive model. That is the bar for bet B3. |

---

## E. New shiro ideas from this analysis

### E1. Encoder-steered online learning ("side-information training"), now bet **B7**
The encoder sees the future and the decoder doesn't. We can still make use of that, cheaply:
```
for each macro-block b:
    encoder tries k training actions a ∈ {replay segment r₁…r_k, LR×{0.5,1,2}, extra step yes/no}
    picks a* = argmin_a  L(blocks b+1 … b+m | θ after action a)       (lookahead, encoder only)
    codes the index of a* with an adaptive model  (≤ log₂ k bits, usually far fewer)
decoder: reads a*, applies the same action → identical θ, no search
```
- **Side-information cost:** about 70k blocks × ~1–2 bits ≈ **10–20 KB**.
- **Encoder cost:** about k× more work. Decoder cost is unchanged. LTCB counts only bytes, so a slower encoder costs us nothing in score.
- This turns replay from a fixed heuristic (B2) into **choices made by an encoder that knows what's coming**.
- **Start with k = 2:** "do the scheduled replay step: yes or no." Then extend.

### E2. Fast and slow logits from one network
Expose three predictions from the same transformer as separate mixer inputs:
- the current weights θ
- an EMA of θ (slow)
- θ plus a tiny LoRA that is reset at every article boundary (fast)

The mixer then learns which time-scale to trust in each context. The cost is one extra forward pass through a small adapter.

---

## F. Priority order for experiments

1. **Free wins that are known to work:** A3 (coder precision), B4 (SSE on all outputs, including the neural one), D6 (inheritance), D7 (dictionary transform).
2. **Cheap tests, possibly large gains:** C1 Muon and C2 schedule-free (raw learning speed), D1 (decay to the prior), B5 multi-candidate match voting.
3. **Our bets:** B1 via D3 (GLN gating on the hidden state), B2 via C5 and C6, **B7 / E1** (encoder-steered training), B3 via D8, B4, B5 via C7.
4. **Architecture research:** C3 and C4 (DeltaNet, Titans memory) as hybrid layers.
