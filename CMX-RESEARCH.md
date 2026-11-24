# cmx research program

The goal is a smaller enwik9 total than the LTCB leaders, starting from their engine. The `cmx/` directory on branch `cmx` holds ex-girlfriend 2.0 (GPL-3, the cmix → cmix-lex → cmix-neif lineage; credits in `cmx/NOTICE.md`). Screens run on free GitHub Actions runners (`.github/workflows/cmx.yml`). Full enwik9 runs (50–95 h per direction) do not fit there.

## How to run an experiment

```sh
# one run per ';'-separated entry: <git ref> [-DNAME=VAL build define ...] [ENV=VAL runtime knob ...]
gh workflow run cmx.yml -f mode=screen -f mb=30 -f label="what this tests" \
  -f runs="cmx <baseline config>;cmx <baseline config> <one change>;..."
gh run download <id> -n summary        # table.md + all.jsonl -> results/gha/cmx-<id>.jsonl
```

- **Stream.** The coder stream is the preprocessed enwik9, 501,062,281 B, sha256 `c35d3073…`. It is identical to upstream's (`mode=stream` rebuilds it into the `cmxstream-v1` cache). A screen codes its first `mb` MB with `comp9 -n`.
- **Cost and time.** 30 MB codes to about 4.94 MB and takes 1.6–3.3 h per runner, depending on which CPU the runner gets. With 20 runners in parallel, 50 MB is the practical maximum under the 6 h job cap.
- **Determinism and noise.** The same config gives the same bytes on every runner (baseline 4,940,203 reproduced 3×). Upstream quotes a ±200 B perturbation floor on 50 MB between near-identical builds, so treat differences under about 200 B as noise.
- **Build defines** (`-DXZ_*`, see `cmx/src/xz_build.h`):
  - SEMCACHE geometry and thresholds: `XZ_SEM_{M,FLIP,NLOG,WHLOG,PRE,DEDUPLOG,L,B,K,G,WTMAX,GAP,TE3,QPE3,SPE3,LRE6}`
  - attention window: `XZ_ATTN_WIN`
- **Runtime knobs.** The `CMX_*` / `BITL_SEQ` knobs are documented in `cmx/src/cmx_meas.h`.
- **Screens under-read memory components.** Upstream got about 20 KB on 50 MB but 378 KB on the full file, because SEMCACHE gains grow as it fills. Rank changes by screen, but never extrapolate them linearly.
- **Memory limit.** The target box (the user's PC) has 16 GB, so the release config must stay under about 15 GB. ex-girlfriend's full recipe needs 24 GB.

## Results so far (30 MB screens, vs cmix-neif 2.0 defaults = 4,940,203 B)

| Config | RSS | Δ |
|---|---|---|
| **best so far:** `-DXZ_SEM_M=31 -DXZ_SEM_FLIP=7 -DXZ_SEM_PRE=512 -DXZ_SEM_DEDUPLOG=10 -DXZ_SEM_GAP=256 -DXZ_SEM_SPE3=985 CMX_SEMKEYS=R,10 CMX_FW_BLOCKS=4 BITL_SEQ=0` | 14.6 GB | **−3,201** |
| ex-girlfriend wide geometry + gap 256 + store 985 + FW4 + BITL0 | 12.0 GB | −3,128 |
| ex-girlfriend wide geometry + FW4 + BITL0 | 12.0 GB | −2,802 |
| ex-girlfriend R,12 cache + FW4 + BITL0 (narrow geometry) | 12.0 GB | −1,943 |
| extra cache layer: 10 −635 · 11 −556 · 13 −525 · 12 −444 · 15 +210 · R,12,9 +147 | | |
| on the wide geometry: gap 256 −152 · store 985 −149 · G=7 −21 · QPE 999 −5 · TE 14/28 +270 | | |
| on the best config (batches 4–5, Δ vs best): `CMX_ROWDECAY=2` −188 · FLIP 11 −177 · gap 512 −85 · gap 128 −50 · G=7 −30 · ROWDECAY=1 −27 · NLOG 25 ±0 · R,11 −2 · R,12 +68 · R,9 +297 | ≤14.9 GB | |
| attention window past the trained 2048 (on best): 2560 +10,431 · 3072 +46,142 · 4096 +51,368 | | dead |
| fxcm context maps ×2 / ×4 (`XZ_CM_MUL`, on defaults): −398 / −542 | +0.7 / +4.9 GB | |
| **batch 9, ONS final stage** `-DXZ_ONS=1 -DXZ_ONS_LAMLOG=20` on best | 14.3 GB | **−549** (biggest single gain so far) |
| ONS λ = 1−2⁻¹⁶: 24.2 MB · λ = 1−2⁻¹³: 38.9 MB · eta 0.5 with λ 2⁻¹⁶: 28.0 MB. **It diverges with fast forgetting**, so it needs a stability guard before any full run | | broken |
| **batch 11, ONS v2** F3 (48 inputs) + EPS 64000 + L20 on BEST2 | 14.4 GB | **4,934,615 = −2,117 vs BEST2 (new best)** |
| batch 10–11 vs BEST2: F3 DMAX 4096 −2,077 · F0 L20 + MA1 + KA900 −2,008 · F0 L20 + KA900 −1,971 · F0 L20 −1,897 · F0 CTX1 −1,885 · F3 DMAX 256 −1,489 · F3 + CTX1 −1,449 · F3 L16/18/22/24 −1,449/−1,421/−1,163/−1,141 · F3 −1,340 · F2 + CTX1 −1,338 · F3 G0 −1,054 · F3 EPS 4000 −961 · F2 −878 · F1 −203. Without ONS: MA1 + KA900 −71, KA 950 +23, KA 800 +3,779 | ≤14.9 GB | |
| batch 10 on BEST (vs 4,937,002): ONS v2 L20 −1,942 (v1 −549) · G16 −2,189 · G0 −2,036 · ETA 1500 −2,036 · DMAX 0 −1,993 · L16 −1,576 · ETA 500 −1,622 · G9 −1,518 · L13 −1,384. Nothing diverges. EPS 1000 and L24: runner died (likely out of memory) | | |
| batch 7 on best: FLIP 11 + ROWDECAY 2 −270 · + gap 512 −227 · gap 512 + RD2 −160 · FLIP 11 + gap 512 −134 · MATCH_AGE=1 −87 · FX_KA=900 −87 · SEMTOP=3 −24 · FWK 0.7 −16 | ≤14.9 GB | |
| batch 7, losers: SEMTOP=1 +26 · KCEN0 +132 · FW 16 blocks +489 (with RD2 +360) · MATCH_AGE=2 +782 · FXLS +1,173 · ADEPTH_TAU 8 +2,434 / 6 +2,869. SEMIN 1 and 2: runner died (~1.8 h) | | dead |
| batch 8, 36 MB vs wide 12 GB (5,939,343): **best −1,167** (the extra cache's lead grows as it fills: −73 at 30 MB) · wide + CM_MUL 2 −620 (14.5 GB) · TAU_CODE 8 −60 · NLOG 26 ±0 · narrow + R,10 + CM_MUL 2 +90 · narrow + R,10 +636 · FW 16 blocks +1,145 · MIXROW 3 +1,300 · Z3LR +1,665 · MIXSKIP_L0 8 +17,137 · NLOG 26 + CM_MUL 2: runner died | | |
| M=47 FLIP=11, and best + CM_MUL=2: the runner died after ~3.7 h without uploading, which looks like running out of memory (>~15 GB) | | |

Raw data is in `results/gha/cmx-*.jsonl`.

## Backlog

Ranked by (expected gain × chance) ÷ cost. The status column tracks each item.

| # | Idea | Source / basis | Status |
|---|---|---|---|
| 1 | Attention window beyond the trained 2048 (2560/3072/4096) at inference | neif-2.0 README §6: 2560 = −13 KB at 39 % on neif-1.0 | dead: +10 KB to +51 KB at 30 MB |
| 2 | Per-row mixer LR decay (`CMX_ROWDECAY=1/2`) | cmix-wolf-2 PRL lever (part of its −28.7 KB) | ROWDECAY=2 −188; stacking in batches 7–8 |
| 3 | Medium geometry × cache layer × gap × ring sizes around the best config | batches 1–3 | done (batch 4); FLIP 11 −177 |
| 4 | Enlarged fxcm text-model tables (memory headroom vs Hutter's 10 GB) | ex-girlfriend 1.0 (not adopted by neif-2.0); lex-transformer: maps halved = +0.11 % | `XZ_CM_MUL` ×2 −398, ×4 −542 on defaults; on the 12 GB wide config in batch 8 |
| 5 | PPMd memory up (+2.2 GB = −0.24 % in lex-transformer) | cmix-lex-transformer writeup | todo |
| 6 | Port cmix-views' models: recurrent word-history state (32-d, ≤16 transitions), extra prediction views with bounded calibration, word-residual correction, cubic disagreement interactions, double-indirect contexts | cmix-views −248 KB vs beehive (part of that is ex-girlfriend 1.0 work we already have) | todo: big port; diff vs beehive is 2,354 lines |
| 7 | RATA-CMIX models: GrammarMatch, MorphologyMatch, MATCHTRUST (8-bit match-outcome history as StateMap context), SPECIALIST (context-gated corrector after SSE) | RATA-CMIX (fx2-cmix-T base, different lineage) | todo: read source |
| 8 | Online 2×200 LSTM expert (RATA) or an online adapter on the frozen transformer (LoRA-style, deterministic CPU) | RATA-CMIX, dynamic evaluation (Krause et al. 2018) | idea |
| 9 | SEMCACHE improvements past "search effort": exact re-rank over more candidates (lift the `kPre <= 512` assertion), cache keyed on attention outputs, kNN over SEMCACHE + match hybrid (RetoMaton pointers) | ex-girlfriend README: the cache is search-limited | idea |
| 10 | Mixer math: Adam/RMS-normalised updates, weight averaging (Polyak), mixture-of-experts gating by transformer confidence | paq8px/cmix lineage; GLN (Veness et al.) | idea |
| 11 | SSE/APM chain: 2D interpolated SSE on (p, transformer entropy), final-stage logistic regression on stretch(p) pairs | paq8px APM chains | idea |
| 12 | Article order v2: order by transformer-embedding similarity (needs a new order block + new stream) | STARLIT/phda ordering; P9 in RESEARCH.md | idea, heavy |
| 14 | Online Newton Step final stage (`-DXZ_ONS=1`, `cmx/src/ons_stage.h`): second-order logistic combiner behind every other stage on [final p, p before/after BITLSTM, end-mixer logit, gate logit, bias], 16 weight sets, A⁻¹ by Sherman–Morrison with forgetting | Hazan et al. 2007 (log loss is exp-concave). Synthetic test: +0.013 % on an already calibrated input, −5 % on a miscalibrated one | v2: −2,117 vs BEST2 (F3, EPS 64k, L20). Stable at every λ tried. Next: fine grid (batch 13) and a length check (batch 12) |
| 13 | Two-pass parameter selection: the encoder tries k knob settings per page and codes the choice (log₂ k bits/page) | P5 "encoder-steered" in RESEARCH.md | idea |

## Log

- 2026-11-23: imported ex-girlfriend 2.0; CI build 9 min; stream reproduced bit-exact; batches 1–3 done (table above); batches 4–5 running.
- 2026-11-24: batches 4–6 done. The best config sits at 14.9 GB and leaves no room on a 16 GB runner, so the overnight batches split. Batch 7 (30 MB, run 45307430312): stacks of FLIP 11, ROWDECAY 2, gap 512 and 16 fast-weight blocks, plus 13 of the screened-and-shelved upstream knobs (SEMIN, SEMTOP, MATCH_AGE, FXLS, ADEPTH_TAU, FX_KA, FWK_MUL, SEM_KCEN0) re-tested on the best config. Batch 8 (36 MB, run 45307436047), on the 12 GB wide config: CM_MUL 2, 2^26 key ring, ROWDECAY, mixer-row and skip knobs, plus the best config itself at 36 MB as a memory test. 36 MB runs past the 2^25 key ring, so this is where NLOG 26 should first show a gain.
- PPMd is built as order 25 with a 14,000 MB pool (`Predictor::AddPPMD`). The pool grows over the full file, so the RSS figures from the screens understate the full-run peak. Check this before picking the release config.
- 2026-11-24: batch 9 (run 45310493581) is 4 variants of the new ONS final stage on the best config (eta 1 / 0.5, lambda 1−2⁻¹³ / 2⁻¹⁶ / 2⁻²⁰). Compare against 4,937,002. It queues behind batches 7–8. Jobs of batches 7–8 that start after commit 1c13702 build the new tree, but with `XZ_ONS` unset the stage is compiled out, so their bytes are unchanged.
- 2026-11-24: batches 7–9 done. New best candidate on paper: best + FLIP 11 + ROWDECAY 2 + ONS(λ 2⁻²⁰), not yet run together (about −820 if the parts add). ONS needs a guard first: its losses at λ 2⁻¹⁶ (+19 MB) show the inverse-Hessian inflation can run away.
- 2026-11-24: ONS v2 (cmx edf91ac). The v1 blow-ups fit non-finite values under `-ffp-model=fast`: `isnan` folds away there, stale `bb_z_` / `m1raw` reach the stage, and a NaN stays in the weights for good. v2 checks floats by bit pattern and adds a ridge cap, a step clamp, a per-set loss guard against its input, and a symmetrized inverse. On synthetic data, v1 goes to +inf% when NaN or inf is injected and v2 to +0.008%; on a calibrated input v2 costs +0.001% (v1 +0.010%). New: `XZ_ONS_FEAT` 1–3 (up to 48 inputs, the BITLSTM's stage-1 row, so a second-order final mixer) and `XZ_ONS_CTX=1` (per-|logit| sets). Batches 10 (45449227899) and 11 (45449236574) are running; batches 12–15 are staged in `CMX-QUEUE.md` for the scheduled agent.
- 2026-11-25: batches 10–11 done. ONS v2 is stable and roughly 4× v1's gain. New best is BEST2 + ONS(F3, EPS 64000, L20) = 4,934,615 (−2,117 vs BEST2). Dispatched batch 12 at 40 MB (run 45510861241; 42 MB risks the job cap on slow runners) and the batch 13 fine grid (run 45510875418).
