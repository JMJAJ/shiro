# cmx run queue

A scheduled agent works through this file every 6 hours. Each time it fires:

1. **Collect.** For every `cmx` run that completed since the last pass, run `gh run download <id> -n summary` and save `all.jsonl` as `results/gha/cmx-<id>.jsonl`. Add the result lines to `CMX-RESEARCH.md` (results table, backlog status, log), mark the batch **done** below with its headline numbers, and commit and push to `main` (no attribution lines).
2. **Dispatch.** If there are no queued or in-progress `cmx` runs, dispatch the next **staged** batches that can start. Use at most 40 jobs, which is two waves of 20. Apply each batch's decision rule to the results so far to fill in its placeholders, then mark it **running** with the run id.
3. **Stop** when nothing is staged. Never use more than 20 runners at once, and never run anything that would push memory past 15 GB (the runner has 16 GB, and a job that runs out kills the runner and loses everything).

Each batch is a `;`-separated list for `gh workflow run cmx.yml -f mode=screen -f mb=<mb> -f label="<label>" -f runs="..."`.

## Names

- `BEST` = `cmx -DXZ_SEM_M=31 -DXZ_SEM_FLIP=7 -DXZ_SEM_PRE=512 -DXZ_SEM_DEDUPLOG=10 -DXZ_SEM_GAP=256 -DXZ_SEM_SPE3=985 <defs> CMX_SEMKEYS=R,10 CMX_FW_BLOCKS=4 BITL_SEQ=0 <env>`. At 30 MB it codes to **4,937,002**; at 36 MB to 5,938,176.
- `BEST2` = `BEST` with `-DXZ_SEM_FLIP=11` and env `CMX_ROWDECAY=2`. At 30 MB it codes to **4,936,732** (−270).
- `BEST3` = `BEST2` with `-DXZ_ONS=1 -DXZ_ONS_LAMLOG=20 -DXZ_ONS_FEAT=3 -DXZ_ONS_EPS=64000` (ONS v2, cmx edf91ac). At 30 MB it codes to **4,934,615** (−2,117 vs BEST2).
- `BEST4` = `BEST3` with `-DXZ_ONS_DMAX=4096 -DXZ_ONS_GUARD=16` (batch 13). At 30 MB it codes to **4,934,288** (−2,444 vs BEST2).
- `BEST5` = `BEST4` with `-DXZ_ONS_ETA=1400` (batch 14). At 30 MB it codes to **4,934,051** (−2,681 vs BEST2).
- `WIDEC` = `cmx -DXZ_SEM_M=63 -DXZ_SEM_FLIP=15 -DXZ_SEM_PRE=512 -DXZ_SEM_DEDUPLOG=10 -DXZ_SEM_GAP=256 -DXZ_SEM_SPE3=985 -DXZ_SEM_NLOG=25 -DXZ_SEM_WHLOG=26 -DXZ_CM_MUL=2 <defs> CMX_FW_BLOCKS=4 BITL_SEQ=0 <env>`. Uses 14.5 GB; at 36 MB it codes to 5,938,723.
- `ONS(...)` = `-DXZ_ONS=1` plus the listed `-DXZ_ONS_*` defines, all on ONS v2 (cmx ≥ 36c9b6c+). `L` = `XZ_ONS_LAMLOG` (λ = 1−2⁻ᴸ), `F` = `XZ_ONS_FEAT`, `C` = `XZ_ONS_CTX`, `E` = `XZ_ONS_ETA`, `G` = `XZ_ONS_GUARD`. ONS v1 (cmx 1c13702) got −549 at L20, and coded the 30 MB to 24–39 MB at L13 and L16. The NaN and inf from fast-math are fixed in v2.
- Timing: 30 MB takes 1.8–3.8 h per job; 42 MB at most about 5.3 h. A job is capped at 355 min, so never screen more than 42 MB.

## Batches

### 10 — ONS v2 ablations (30 MB) — done, run 45449227899
Result: v2 fixes v1. BEST + ONS L20 = 4,935,060 (−1,942 vs BEST; v1 got −549). Best result: BEST2 + ONS L20 + MATCH_AGE 1 + FX_KA 900 = 4,934,724 (−2,008 vs BEST2). Best ablation: GUARD 16 (on BEST −247 vs default guard 12). GUARD 9 is +424 and L16 / L13 are +366 / +558, and none of them diverge. Without ONS, MATCH_AGE + FX_KA on BEST2 = −71. EPS 1000 and L24: runner died (shutdown about 1 h in, likely out of memory).
Tests v2 against v1's −549, plus λ, guard, ridge, step size, and the stack with BEST2 / MATCH_AGE / FX_KA.

### 11 — ONS input width and context (30 MB) — done, run 45449236574
Result: **F3 + EPS 64000 L20 on BEST2 = 4,934,615 (−2,117 vs BEST2, new best)**. F3 + DMAX 4096 = −2,077. F3 alone (−1,340) is worse than F0 (−1,897) unless it gets more ridge or a looser step clamp. CTX 1 is ±0 on F0 and +109 on F3. F1 −203, F2 −878. L16/18/22/24 on F3 are all worse than L20. WIDEC + ONS F3 at 30 MB = 4,936,385.
Tests FEAT 1/2/3 and CTX 1 on BEST2, λ around 20, EPS / DMAX for the 48-input version, and WIDEC + ONS.

### 12 — winners at length (40 MB) — done, run 45510861241
Result (vs BEST2 at 40 MB = 6,537,921): F3 DMAX4096 L20 −2,298 (−2,077 at 30 MB), F0 L20 + MA1 + KA900 −2,296 (−2,008), F0 L20 + KA900 −2,254, F0 L20 −2,082. The ONS gain grows a little with length. L22 is worse than L20 on all four (−2,128 to −2,237). BEST +384 vs BEST2. WIDEC +1,585; WIDEC + ONS F3 +845, so BEST2 still leads WIDEC. #3 (F3 EPS64000 L20): runner died at ~3.2 h (likely out of memory). #17 (BEST + G0): timed out (exit 124) on a slow runner.
Filled in: #0 BEST2, #1 BEST, #2 WIDEC. #3–6: F3 EPS64000 L20, F3 DMAX4096 L20, F0 L20 + MA1 + KA900, and F0 L20 + KA900 (all on BEST2). #7–10: the same four at L22. #11: WIDEC + F3 EPS64000 L20. #12–19 are the next-best 30 MB configs: BEST + G16, BEST2 + F0 L20, CTX 1, SEMTOP 3, MA 1, BEST + G0, BEST + ETA 1500, BEST + DMAX 0. Screened at 40 MB, not 42 MB, because the slowest 30 MB jobs take 15,000 s and 42 MB would hit the 355 min cap.
Rule: take the 4 best configs from batches 10–11 (ONS variants on BEST2, and anything else under BEST2 − 150). Run those 4 as written; each of those 4 with L one step slower (L+2); BEST, BEST2 and WIDEC as baselines at 42 MB; and WIDEC with the best ONS setting. That's up to 12 runs. Fill to 20 with the next-best 30 MB configs. Purpose: check whether the ONS gain grows with length (its regret is log T) and whether BEST still leads WIDEC.

### 13 — ONS fine grid (30 MB) — done, run 45510875418
Result (vs anchor BEST3 = 4,934,615): **DMAX 4096 + G16 = 4,934,288 (−327; −2,444 vs BEST2) = BEST4**. G16 −318, G14 −235, G10 +294; ETA 1400 −141, ETA 700 +321; L19 −15, L21 +57; EPS 128k/256k −16/−13, 32k +259; DMAX 2048/4096/0 about ±0, 512 +183; CTX 1 +211; F2 + C1 +221; F0 + EPS64k +183; F0 + G16 −19.
Filled in around F3 EPS64000 L20 on BEST2 (#0 rerun as the anchor): L19/21, ETA 700/1400, DMAX 512/2048, G10/14, EPS 32k/128k/256k (16k is the known 4,935,392), C1, F2 + C1 + EPS64k, DMAX 4096, G16, DMAX 4096 + G16, DMAX 0, F0 + EPS64k, F0 + G16.
Rule: around the best ONS setting `(L, F, C, E)`, run L ±1, E ×0.7/×1.4, DMAX ½/2, G 10/14, EPS ¼/4, and C flipped. If F3 won, also try F2 with C1. That's 20 runs.

### 14 — ONS × the knobs that lost on their own (30 MB) — done, run 45611575846
Result (vs BEST4 = 4,934,288): **ETA 1400 = 4,934,051 (−237; −2,681 vs BEST2) = BEST5**. MA1 + KA900 −79, KA900 −20, G18 +1, G20 +14, SEMTOP 3 +31, GAP 512 +42, FWK 0.7 +42, R,11 +51, MA1 +77, R,12 +135. G18 + ETA 1400: runner died at ~1.8 h (likely out of memory). Only ETA 1400 and the MA1 + KA900 pair survive under ONS.
Filled in on BEST4 (no anchor rerun; compare to 4,934,288): #0 MA1, #1 KA900, #2 MA1 + KA900, #3 SEMTOP 3, #4 GAP 512, #5 FWK_MUL 0.7, #6 R,11, #7 R,12. Extra grid points from batch 13's trend: #8 ETA 1400, #9 G18, #10 G20, #11 G18 + ETA 1400. 12 jobs.
Rule: with the best ONS setting on BEST2, add one at a time `CMX_MATCH_AGE=1`, `CMX_FX_KA=900`, `CMX_SEMTOP=3`, `-DXZ_SEM_GAP=512`, `CMX_FWK_MUL=0.7`, and `CMX_SEMKEYS=R,11` / `R,12`. Pairs are only for winners from batch 10. Purpose: a final stage can turn a slightly worse but differently-wrong upstream into a gain.

### 15 — release candidate at 40 MB, three seeds of the stream offset — done, runs 45672750456 (40 MB) and 45672775112 (20 MB)
Result (vs BEST2): BEST5 at 20 / 30 / 40 MB = −1,848 / −2,681 / −2,985 (0.055 / 0.054 / 0.046 % of BEST2's bytes). At 40 MB: BEST5 + MA1 + KA900 = 6,534,819 (−3,102, best), BEST5 6,534,936, BEST4 + MA1 + KA900 6,535,081 (−2,840). At 20 MB BEST4 + MA1 + KA900 −1,707. Summary for the user is in `CMX-RESEARCH.md`.
Filled in: best = BEST5, runner-up = BEST4 + MA1 + KA900 (4,934,209), plus one untested stack BEST5 + MA1 + KA900. 40 MB run: #0 BEST5, #1 BEST4 + MA1 + KA900, #2 BEST5 + MA1 + KA900 (BEST2 at 40 MB is known: 6,537,921). 20 MB run: #0 BEST2, #1 BEST5, #2 BEST4 + MA1 + KA900. 30 MB bytes for all three are known, so no 30 MB reruns. 40 MB, not 42 MB: batch 12's 40 MB jobs took up to 20,198 s (337 min) against the 355 min cap; RSS at 40 MB peaked at 14.9 GB. 6 jobs. When done: write the summary below for the user and stop.
Rule: the single best stack from batches 12–14, plus BEST2, plus the runner-up stack, each at 42 MB. There is no stream-offset knob, so instead also run each at 20 MB and 30 MB to fit the gain-versus-length curve. That's 9 runs, then stop and leave a summary in `CMX-RESEARCH.md` for the user: the predicted full-file gain, with the warning that screens under-read gains that build up over the file.

## Port track (from 2026-11-25): ideas from other top-10 entries

The user asked to prioritise combining mechanisms from other entries over more configuration sweeps. Sources are cloned read-only under `../ports/`. Each port sits behind its own build define on `cmx`, is screened as an add-on to BEST5 (and to BEST2, to see whether it overlaps with ONS), and only winners get a second round. wolf-2 has nothing new: its E1 and E3b levers are already in neif-2.0. The overlap question is the research value: a port that wins on BEST2 but not on BEST5 is information ONS already extracts.

Defines (cmx ≥ 17ec53d):
- `XZ_WRES=1` — cmix-views' word-context residual (`src/wres_stage.h`). `XZ_WRES_LR0` is the starting gate log-odds ×100 (cmix-views 0; on synthetic data −200 or −400 cut the cost on clean input from +0.065 % to +0.012 % / +0.002 %). `XZ_WRES_W` is the number of key words, 1–3. `XZ_WRES_CELLLOG` is the cell cap, default 20.
- `XZ_WGRU=1` — cmix-views' recurrent word history (`src/word_gru.h` inside `word_label.cpp`, plus `src/wgru_stage.h`). About +5–10 % time.
- `XZ_MTRUST=1|2` — RATA's MATCHTRUST (`models/fxcmv1.cpp`). 1 = 4 history bits in the match length context; 2 = RATA's 4th StateMap (+2 fxcm outputs, +16 input lanes).

### P1 — word residual (30 MB) — done, run 45681545454
Result: loses everywhere. Best is BEST5 + LR0 −400 = 4,934,054 (+3 vs BEST5); LR0 −200 W1 +27, LR0 −200 +51, CELLLOG 21 +68, W2 +128, LR0 0 +436. On BEST2: LR0 0 +436, LR0 −200 +52. The more negative the gate start, the closer to ±0: the stage only costs.
#0 BEST5+WRES · #1 LR0 −200 · #2 LR0 −400 · #3 LR0 −200 W2 · #4 LR0 −200 W1 · #5 BEST2+WRES · #6 BEST2+WRES LR0 −200 · #7 BEST5 LR0 −200 CELLLOG 21. Compare against BEST5 4,934,051 and BEST2 4,936,732.

### P2 — word GRU (30 MB) — done, run 45682779594
Result: ±0. BEST5 + WGRU 4,934,049 (−2), BEST2 + WGRU 4,936,729 (−3); with WRES(LR0 −200) on BEST5 +48, with WRES on BEST2 +432.
#0 BEST5+WGRU · #1 BEST2+WGRU · #2 BEST5+WGRU+WRES(LR0 −200) · #3 BEST2+WGRU+WRES.

### P3 — MATCHTRUST (30 MB) — done, run 45683305363
Result: MTRUST1 on BEST5 = 4,934,027 (−24), on BEST2 4,936,639 (−93): a small gain, mostly absorbed by ONS, under the 150 B bar. MTRUST2 (#1 BEST5, #3 BEST2): runner died, no artifact (likely out of memory; +16 input lanes).
#0 BEST5+MTRUST1 · #1 BEST5+MTRUST2 · #2 BEST2+MTRUST1 · #3 BEST2+MTRUST2.

### P4 — combination round (30 MB) — skipped (no winner)
No port setting beat BEST5 by 150 B (best: MTRUST1 −24), so per the rule nothing ran.
Rule: the winners are each port's best setting at least 150 B under BEST5. Run every pair and the triple of winners on BEST5, each winner's best setting with one step either side (for example LR0 ±200, or MTRUST 1 vs 2 if both won), and each winner on BEST5 + MA1 + KA900. At most 20 jobs. If nothing beats BEST5 by 150 B, run nothing, and say in `CMX-RESEARCH.md` that the port overlaps with ONS.

### P5 — the port stack at length (40 MB) — skipped (no P4 stack)
Port-track summary is in `CMX-RESEARCH.md`.
Rule: the best P4 stack and BEST5 at 40 MB (BEST5's 40 MB number may already be known from batch 15), plus the best stack at 20 MB. That's 3 jobs. Then write a port-track summary in `CMX-RESEARCH.md`: each port's gain on BEST2 vs on BEST5 (overlap with ONS), the stack, and its growth with length.

Ports that still need code (the scheduled runner skips them): cmix-views' view experts (4 online output heads on sparse transformer-block activations, `tf1/opt/view_expert.h`), RATA's MorphologyMatch, Scr2Match and CausalDonor, and cmix-lex-transformer's larger PPMd (memory).

## Not for this queue (needs the user)
- An encode → decode round trip of ONS (`-e` / `-d` on a few MB) before any release use. The screens only run the encoder.
- The full-file PPMd pool growth (14,000 MB cap) against the 16 GB box.
