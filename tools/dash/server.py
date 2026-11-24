#!/usr/bin/env python3
"""shiro lab: a local dashboard for runs, live jobs on the test box, and a formula lab.

    SHIRO_PASS=... python tools/dash/server.py          # http://127.0.0.1:8765
    python tools/dash/server.py --port 9000 --no-remote

Env: SHIRO_HOST / SHIRO_USER / SHIRO_PASS (as tools/remote.py; the password is only held
in memory), SHIRO_DIRS = remote checkouts to read (default "xzip,xzip-dev,xzip-dev2").
Binds to 127.0.0.1 only. Formulas and journal notes are saved in tools/dash/lab.json.
"""
import argparse
import json
import os
import shlex
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

LAB_FILE = os.path.join(HERE, "lab.json")
LOCAL_RUNS = os.path.join(ROOT, "results", "runs.jsonl")
DIRS = [d for d in os.environ.get("SHIRO_DIRS", "xzip,xzip-dev,xzip-dev2").split(",") if d]


# ---------------- remote (SSH) ----------------

class Remote:
    def __init__(self, enabled):
        self.enabled = enabled and bool(os.environ.get("SHIRO_PASS") or os.environ.get("SHIRO_KEYS"))
        self.lock = threading.Lock()
        self.client = None
        self.error = None if self.enabled else "remote disabled (set SHIRO_PASS)"

    def _conn(self):
        if self.client is not None:
            t = self.client.get_transport()
            if t is not None and t.is_active():
                return self.client
        import remote  # tools/remote.py
        self.client = remote.connect()
        return self.client

    def run(self, cmd, timeout=20):
        if not self.enabled:
            raise RuntimeError(self.error)
        with self.lock:
            try:
                c = self._conn()
                _, out, _ = c.exec_command(cmd, timeout=timeout)
                data = out.read().decode(errors="replace")
                self.error = None
                return data
            except Exception as e:  # reconnect next time
                self.client = None
                self.error = f"{type(e).__name__}: {e}"
                raise

    def read(self, path):
        if not self.enabled:
            raise RuntimeError(self.error)
        with self.lock:
            try:
                c = self._conn()
                sftp = c.open_sftp()
                try:
                    with sftp.open(path, "r") as f:
                        return f.read().decode(errors="replace")
                finally:
                    sftp.close()
            except FileNotFoundError:
                return ""
            except Exception as e:
                self.client = None
                self.error = f"{type(e).__name__}: {e}"
                raise


REMOTE = None
CACHE = {}
CACHE_LOCK = threading.Lock()


def cached(key, ttl, fn):
    now = time.time()
    with CACHE_LOCK:
        hit = CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with CACHE_LOCK:
        CACHE[key] = (now, val)
    return val


# ---------------- runs ----------------

def classify(r):
    if r.get("kind") == "nn":
        return "nn"
    if str(r.get("note", "")).startswith("hybrid"):
        return "hybrid"
    return "cm"


def parse_jsonl(text, source):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        r["source"] = source
        r["kind"] = classify(r)
        out.append(r)
    return out


def load_runs(refresh):
    runs = []
    if os.path.exists(LOCAL_RUNS):
        runs += parse_jsonl(open(LOCAL_RUNS, encoding="utf-8").read(), "local")
    errors = []
    if REMOTE.enabled:
        def fetch():
            got = []
            for d in DIRS:
                try:
                    got += parse_jsonl(REMOTE.read(f"{d}/results/runs.jsonl"), d)
                except Exception as e:
                    errors.append(f"{d}: {e}")
            return got
        if refresh:
            with CACHE_LOCK:
                CACHE.pop("remote_runs", None)
        runs += cached("remote_runs", 60, fetch)
    # dedupe (the same run can be in the local copy and on the box)
    seen, uniq = set(), []
    for r in runs:
        k = (r.get("ts"), r.get("kind"), r.get("file"), r.get("out"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: r.get("ts", 0))
    return {"runs": uniq, "errors": errors, "remote": REMOTE.enabled}


# ---------------- live status ----------------

STATUS_CMD = r"""
echo '##uptime'; uptime
echo '##mem'; free -m | awk 'NR==2{print $2, $3, $7}'
echo '##gpu'; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null
echo '##cpu'; top -bn1 | awk -F'[ ,]+' '/^%Cpu/{for(i=1;i<=NF;i++) if($i=="id") print 100-$(i-1)}'
echo '##procs'; ps -eo pid,etimes,pcpu,rss,args --sort=-pcpu | grep -E 'release/[x]zip|nn/[o]nline|[h]ybrid.sh|[s]weep.sh|[n]nsweep.sh' | head -20
echo '##logs'
for d in DIRS; do
  for f in $(ls -t ~/$d/logs/*.log 2>/dev/null | head -8); do
    echo "$d|$(basename $f .log)|$(stat -c %Y $f)|$(tail -c 400 $f | tr '\r' '\n' | grep -av '^\s*$' | tail -1)"
  done
done
echo '##now'; date +%s
"""


def load_status():
    cmd = STATUS_CMD.replace("DIRS", " ".join(shlex.quote(d) for d in DIRS))
    raw = REMOTE.run(cmd)
    sec, cur = {}, None
    for line in raw.splitlines():
        if line.startswith("##"):
            cur = line[2:]
            sec[cur] = []
        elif cur:
            sec[cur].append(line)
    st = {"uptime": " ".join(sec.get("uptime", [])).strip()}
    m = (sec.get("mem") or [""])[0].split()
    if len(m) == 3:
        st["mem"] = {"total": int(m[0]), "used": int(m[1]), "avail": int(m[2])}
    g = (sec.get("gpu") or [""])[0].split(", ")
    if len(g) >= 6:
        st["gpu"] = {"name": g[0], "util": float(g[1]), "mem_used": float(g[2]), "mem_total": float(g[3]),
                     "temp": float(g[4]), "power": g[5]}
    c = sec.get("cpu") or []
    try:
        st["cpu"] = float(c[0])
    except (IndexError, ValueError):
        pass
    procs = []
    for line in sec.get("procs", []):
        p = line.split(None, 4)
        if len(p) == 5 and "##procs" not in p[4] and not p[4].startswith("grep"):
            procs.append({"pid": int(p[0]), "etimes": int(p[1]), "pcpu": float(p[2]), "rss_mb": int(p[3]) // 1024,
                          "args": p[4]})
    st["procs"] = procs
    logs = []
    for line in sec.get("logs", []):
        p = line.split("|", 3)
        if len(p) == 4:
            logs.append({"dir": p[0], "name": p[1], "mtime": int(p[2]), "last": p[3]})
    logs.sort(key=lambda x: -x["mtime"])
    st["logs"] = logs
    try:
        st["now"] = int(sec["now"][0])
    except (KeyError, IndexError, ValueError):
        st["now"] = int(time.time())
    return st


def load_log(d, name, n):
    if d not in DIRS or "/" in name or ".." in name:
        raise ValueError("bad log")
    return REMOTE.run(f"tail -n {int(n)} ~/{shlex.quote(d)}/logs/{shlex.quote(name)}.log | tr '\\r' '\\n' | grep -av '^\\s*$' | tail -n {int(n)}")


# ---------------- lab (formulas + journal) ----------------

LAB_LOCK = threading.Lock()


def load_lab():
    if os.path.exists(LAB_FILE):
        return json.load(open(LAB_FILE, encoding="utf-8"))
    return {"formulas": [], "notes": []}


def save_lab(obj):
    with LAB_LOCK:
        tmp = LAB_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, LAB_FILE)


# ---------------- http ----------------

class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/api/runs":
                self._send(200, load_runs(q.get("refresh") == "1"))
            elif u.path == "/api/status":
                if not REMOTE.enabled:
                    self._send(200, {"error": REMOTE.error})
                else:
                    try:
                        self._send(200, cached("status", 4, load_status))
                    except Exception as e:
                        self._send(200, {"error": f"{type(e).__name__}: {e}"})
            elif u.path == "/api/log":
                self._send(200, {"text": load_log(q.get("dir", ""), q.get("name", ""), q.get("n", "120"))})
            elif u.path == "/api/lab":
                self._send(200, load_lab())
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_PUT(self):
        u = urlparse(self.path)
        # only same-origin requests from the page itself (blocks drive-by cross-site writes)
        if self.headers.get("X-Lab") != "1":
            return self._send(403, {"error": "forbidden"})
        if u.path != "/api/lab":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        try:
            obj = json.loads(self.rfile.read(n))
            assert isinstance(obj.get("formulas"), list) and isinstance(obj.get("notes"), list)
        except Exception:
            return self._send(400, {"error": "bad lab json"})
        save_lab(obj)
        self._send(200, {"ok": True})


def main():
    global REMOTE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-remote", action="store_true")
    a = ap.parse_args()
    REMOTE = Remote(not a.no_remote)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"shiro lab on http://127.0.0.1:{a.port}  (remote: {'on' if REMOTE.enabled else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
