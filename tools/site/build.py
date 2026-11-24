#!/usr/bin/env python3
"""Build the GitHub Pages site: site/ (static) + data.json from results and Actions runs.

    python3 tools/site/build.py --out _site [--meta runs_meta.json] [--fetched DIR]
                                [--stream-rate stream_rate.tsv] [--jobs jobs.json]

--meta     gh run list JSON for the sweep / cmx / enwik9 workflows (titles, states)
--fetched  directory of downloaded `summary` artifacts: <DIR>/<run id>/all.jsonl
--jobs     {run id: [job {name,status,conclusion,started_at}]} for runs in progress
"""
import argparse
import glob
import json
import os
import re
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LTCB enwik9 entries the program measures itself against (archive9 bytes as published
# by each entry; the ranking total also counts the decompressor).
LTCB = [
    ("ex-girlfriend 2.0", 88_983_124),
    ("cmix-neif 2.0", 89_346_392),
    ("cmix-views 1.0", 90_515_901),
    ("cmix-wolf-2 P14", 90_735_088),
    ("cmix-neif-beehive", 90_763_792),
    ("makimono 1.0", 95_793_872),
    ("cmix-lex-transformer", 95_836_613),
    ("RATA-CMIX", 96_849_689),
    ("fx2-cmix-T-ft8 (#10)", 96_945_005),
]


def read_jsonl(path):
    out = []
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


def tokens(r):
    """The configuration of one run as tokens: build defines and env knobs split on
    whitespace (values such as R,12 keep their commas), CM tune specs on commas."""
    t = []
    for k in ("defs", "env"):
        t += (r.get(k) or "").split()
    t += [x for x in (r.get("tune") or "").split(",") if x.strip()]
    if r.get("bits") and "env" not in r:
        t.append(f"bits={r['bits']}")
    return t


WIDE = {"-DXZ_SEM_M=63", "-DXZ_SEM_FLIP=15", "-DXZ_SEM_NLOG=25", "-DXZ_SEM_WHLOG=26", "-DXZ_SEM_PRE=512",
        "-DXZ_SEM_DEDUPLOG=10"}


def short(tok):
    return tok.replace("-DXZ_SEM_", "sem.").replace("-DXZ_", "").replace("CMX_", "")


def sweep_rows(rs):
    rs = sorted(rs, key=lambda r: r.get("i", 0))
    rs = [r for r in rs if r.get("out")]
    base = next((r for r in rs if r.get("i") == 0), None)
    toks = [tokens(r) for r in rs]
    common = set.intersection(*(set(t) for t in toks)) if toks else set()
    refs = {r.get("ref") for r in rs}
    rows = []
    for r, tk in zip(rs, toks):
        own = [x for x in tk if x not in common]
        if WIDE <= set(own):   # ex-girlfriend 2.0's wider SEMCACHE search, as one token
            own = ["sem.wide"] + [x for x in own if x not in WIDE]
        own = [short(x) for x in own]
        if len(refs) > 1:
            own.insert(0, r.get("ref", ""))
        label = " ".join(own) or ("baseline" if r is base else "same as #0")
        d = (r["out"] - base["out"]) if base else None
        rows.append({
            "i": r.get("i"), "label": label, "config": " ".join([r.get("ref", "")] + tk), "bytes": int(r["out"]),
            "delta": d, "pct": (d * 100 / base["out"]) if base and d is not None else None,
            "secs": r.get("secs"), "rss": r.get("rss_mb"), "cpu": r.get("cpu"), "rev": r.get("rev"),
        })
    return rows, (base["out"] if base else None), " ".join(short(x) for x in toks[0] if x in common) if toks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--fetched", default=None)
    ap.add_argument("--stream-rate", default=None)
    ap.add_argument("--jobs", default=None)
    a = ap.parse_args()

    meta = {}
    if a.meta and os.path.exists(a.meta):
        for m in json.load(open(a.meta)):
            meta[str(m["databaseId"])] = m

    # sweeps: committed results first, then summaries fetched from Actions
    files = {}
    for p in glob.glob(os.path.join(ROOT, "results", "gha", "*.jsonl")):
        rid = os.path.basename(p)[:-6].replace("cmx-", "")
        files[rid] = p
    if a.fetched:
        for p in glob.glob(os.path.join(a.fetched, "*", "all.jsonl")):
            rid = os.path.basename(os.path.dirname(p))
            files.setdefault(rid, p)

    sweeps = []
    for rid, p in files.items():
        rs = read_jsonl(p)
        if not rs:
            continue
        rows, base, shared = sweep_rows(rs)
        if not rows:
            continue
        m = meta.get(rid, {})
        is_cmx = "mb" in rs[0] or "env" in rs[0]
        title = m.get("displayTitle") or ""
        sweeps.append({
            "id": rid, "kind": "cmx" if is_cmx else "cm",
            "title": title or (f"cmx {rs[0].get('mb')} MB" if is_cmx else f"{rs[0].get('file', '')}"),
            "created": m.get("createdAt") or "", "url": m.get("url") or f"https://github.com/JMJAJ/shiro/actions/runs/{rid}",
            "file": rs[0].get("file", ""), "mb": rs[0].get("mb"), "base": base, "shared": shared, "rows": rows,
        })
    sweeps.sort(key=lambda s: (s["created"] or "", s["id"]))

    # long runs (enwik8/9 evals, NN runs) from the run log
    runs = []
    for r in read_jsonl(os.path.join(ROOT, "results", "runs.jsonl")):
        if not r.get("out") or not r.get("in"):
            continue
        runs.append({
            "ts": r.get("ts"), "file": os.path.basename(r.get("file", "")), "in": r["in"], "out": r["out"],
            "bpc": r.get("bpc"), "kind": r.get("kind", "cm"), "note": (r.get("note") or "")[:80],
            "secs": r.get("secs"), "regions": r.get("regions"),
        })

    stream = []
    if a.stream_rate and os.path.exists(a.stream_rate):
        for line in open(a.stream_rate):
            if line[:1].isdigit():
                f0, f1, ob, bpb = line.split("\t")[:4]
                stream.append([round((int(f0) + int(f1)) / 2e6, 2), float(bpb)])

    live = []
    jobs = json.load(open(a.jobs)) if a.jobs and os.path.exists(a.jobs) else {}
    for rid, m in meta.items():
        if m.get("status") in ("in_progress", "queued", "waiting", "pending"):
            js = jobs.get(rid, [])
            live.append({
                "id": rid, "title": m.get("displayTitle"), "workflow": m.get("workflowName"), "status": m.get("status"),
                "created": m.get("createdAt"), "url": m.get("url"),
                "jobs": [{"name": j.get("name"), "status": j.get("status"), "conclusion": j.get("conclusion"),
                          "started": j.get("started_at")} for j in js],
            })
    live.sort(key=lambda x: x["created"] or "")

    formulas = json.load(open(os.path.join(ROOT, "tools", "site", "formulas.json")))

    data = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": "JMJAJ/shiro",
        "ltcb": [{"name": n, "bytes": b} for n, b in LTCB],
        "sweeps": sweeps, "runs": runs, "stream_rate": stream, "live": live, "formulas": formulas,
    }
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, "site")):
        shutil.copy(os.path.join(ROOT, "site", f), os.path.join(a.out, f))
    with open(os.path.join(a.out, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"site: {len(sweeps)} sweeps, {len(runs)} runs, {len(stream)} stream points, {len(live)} live")


if __name__ == "__main__":
    main()
