# cmx run queue

A scheduled agent works through this file every 6 hours. Each time it fires:

1. **Collect.** For every `cmx` run that completed since the last pass, run `gh run download <id> -n summary` and save `all.jsonl` as `results/gha/cmx-<id>.jsonl`. Add the result lines to `CMX-RESEARCH.md` (results table, backlog status, log), mark the batch **done** below with its headline numbers, and commit and push to `main` (no attribution lines).
2. **Dispatch.** If there are no queued or in-progress `cmx` runs, dispatch the next **staged** batches that can start. Use at most 40 jobs, which is two waves of 20. Apply each batch's decision rule to the results so far to fill in its placeholders, then mark it **running** with the run id.
3. **Stop** when nothing is staged. Never use more than 20 runners at once, and never run anything that would push memory past 15 GB (the runner has 16 GB, and a job that runs out kills the runner and loses everything).

Each batch is a `;`-separated list for `gh workflow run cmx.yml -f mode=screen -f mb=<mb> -f label="<label>" -f runs="..."`.

## Names

- `BEST` = `cmx -DXZ_SEM_M=31 -DXZ_SEM_FLIP=7 -DXZ_SEM_PRE=512 -DXZ_SEM_DEDUPLOG=10 -DXZ_SEM_GAP=256 -DXZ_SEM_SPE3=985 <defs> CMX_SEMKEYS=R,10 CMX_FW_BLOCKS=4 BITL_SEQ=0 <env>`. At 30 MB it codes to **4,937,002**; at 36 MB to 5,938,176.
- `BEST2` = `BEST` with `-DXZ_SEM_FLIP=11` and env `CMX_ROWDECAY=2`. At 30 MB it codes to **4,936,732** (−270).
- `WIDEC` = `cmx -DXZ_SEM_M=63 -DXZ_SEM_FLIP=15 -DXZ_SEM_PRE=512 -DXZ_SEM_DEDUPLOG=10 -DXZ_SEM_GAP=256 -DXZ_SEM_SPE3=985 -DXZ_SEM_NLOG=25 -DXZ_SEM_WHLOG=26 -DXZ_CM_MUL=2 <defs> CMX_FW_BLOCKS=4 BITL_SEQ=0 <env>`. Uses 14.5 GB; at 36 MB it codes to 5,938,723.
- `ONS(...)` = `-DXZ_ONS=1` plus the listed `-DXZ_ONS_*` defines, all on ONS v2 (cmx ≥ 36c9b6c+). `L` = `XZ_ONS_LAMLOG` (λ = 1−2⁻ᴸ), `F` = `XZ_ONS_FEAT`, `C` = `XZ_ONS_CTX`, `E` = `XZ_ONS_ETA`, `G` = `XZ_ONS_GUARD`. ONS v1 (cmx 1c13702) got −549 at L20, and coded the 30 MB to 24–39 MB at L13 and L16. The NaN and inf from fast-math are fixed in v2.
- Timing: 30 MB takes 1.8–3.8 h per job; 42 MB at most about 5.3 h. A job is capped at 355 min, so never screen more than 42 MB.

## Batches

### 10 — ONS v2 ablations (30 MB) — running, run 45449227899
Tests v2 against v1's −549, plus λ, guard, ridge, step size, and the stack with BEST2 / MATCH_AGE / FX_KA.

### 11 — ONS input width and context (30 MB) — running, run 45449236574
Tests FEAT 1/2/3 and CTX 1 on BEST2, λ around 20, EPS / DMAX for the 48-input version, and WIDEC + ONS.

### 12 — winners at length (42 MB) — staged, after 10 and 11
Rule: take the 4 best configs from batches 10–11 (ONS variants on BEST2, and anything else under BEST2 − 150). Run those 4 as written; each of those 4 with L one step slower (L+2); BEST, BEST2 and WIDEC as baselines at 42 MB; and WIDEC with the best ONS setting. That's up to 12 runs. Fill to 20 with the next-best 30 MB configs. Purpose: check whether the ONS gain grows with length (its regret is log T) and whether BEST still leads WIDEC.

### 13 — ONS fine grid (30 MB) — staged, after 10 and 11
Rule: around the best ONS setting `(L, F, C, E)`, run L ±1, E ×0.7/×1.4, DMAX ½/2, G 10/14, EPS ¼/4, and C flipped. If F3 won, also try F2 with C1. That's 20 runs.

### 14 — ONS × the knobs that lost on their own (30 MB) — staged, after 13
Rule: with the best ONS setting on BEST2, add one at a time `CMX_MATCH_AGE=1`, `CMX_FX_KA=900`, `CMX_SEMTOP=3`, `-DXZ_SEM_GAP=512`, `CMX_FWK_MUL=0.7`, and `CMX_SEMKEYS=R,11` / `R,12`. Pairs are only for winners from batch 10. Purpose: a final stage can turn a slightly worse but differently-wrong upstream into a gain.

### 15 — release candidate at 42 MB, three seeds of the stream offset — staged, after 12 and 14
Rule: the single best stack from batches 12–14, plus BEST2, plus the runner-up stack, each at 42 MB. There is no stream-offset knob, so instead also run each at 20 MB and 30 MB to fit the gain-versus-length curve. That's 9 runs, then stop and leave a summary in `CMX-RESEARCH.md` for the user: the predicted full-file gain, with the warning that screens under-read gains that build up over the file.

## Not for this queue (needs the user)
- An encode → decode round trip of ONS (`-e` / `-d` on a few MB) before any release use. The screens only run the encoder.
- The full-file PPMd pool growth (14,000 MB cap) against the 16 GB box.
