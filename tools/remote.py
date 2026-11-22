#!/usr/bin/env python3
"""Sync this repo to the test box and run commands there.

    python tools/remote.py sync                 # push source tree (no target/, data/, .git/)
    python tools/remote.py run "<shell cmd>"    # run in ~/xzip on the box
    python tools/remote.py bg  <name> "<cmd>"   # run detached, log to ~/xzip/logs/<name>.log
    python tools/remote.py log <name> [lines]   # tail that log
    python tools/remote.py get <remote> <local> # fetch a file

Connection comes from env: XZIP_HOST, XZIP_USER, XZIP_PASS, XZIP_DIR (default xzip) (password is never
written to disk). Requires paramiko.
"""
import io
import os
import subprocess
import sys
import tarfile

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_DIR = os.environ.get("XZIP_DIR", "xzip")
EXCLUDE_DIRS = {"target", "data", ".git", "__pycache__", "logs"}


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        os.environ.get("XZIP_HOST", "192.168.0.116"),
        username=os.environ.get("XZIP_USER", "Jxint"),
        password=os.environ.get("XZIP_PASS"),
        timeout=15,
        allow_agent=os.environ.get("XZIP_PASS") is None,
        look_for_keys=os.environ.get("XZIP_PASS") is None,
    )
    return c


def run(c, cmd, stream=True):
    full = f"cd ~/{REMOTE_DIR} && source ~/.cargo/env 2>/dev/null; {cmd}"
    chan = c.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(full)
    out = []
    while True:
        data = chan.recv(65536)
        if not data:
            break
        s = data.decode(errors="replace")
        out.append(s)
        if stream:
            sys.stdout.write(s)
            sys.stdout.flush()
    return chan.recv_exit_status(), "".join(out)


def rev():
    try:
        sha = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", ROOT, "diff", "--quiet", "HEAD"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def sync(c):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                p = os.path.join(dirpath, f)
                tar.add(p, arcname=os.path.relpath(p, ROOT).replace(os.sep, "/"))
        r = rev().encode()
        ti = tarfile.TarInfo(".rev")
        ti.size = len(r)
        tar.addfile(ti, io.BytesIO(r))
    buf.seek(0)
    sftp = c.open_sftp()
    try:
        sftp.mkdir(REMOTE_DIR)
    except IOError:
        pass
    sftp.putfo(buf, f"{REMOTE_DIR}/.sync.tgz")
    sftp.close()
    code, _ = run(c, "tar xzf .sync.tgz && rm .sync.tgz && mkdir -p logs results && echo synced $(cat .rev)")
    return code


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    c = connect()
    op = sys.argv[1]
    if op == "sync":
        sys.exit(sync(c))
    elif op == "run":
        code, _ = run(c, sys.argv[2])
        sys.exit(code)
    elif op == "bg":
        name, cmd = sys.argv[2], sys.argv[3]
        code, _ = run(c, f"mkdir -p logs; nohup bash -c {sh_quote(cmd)} > logs/{name}.log 2>&1 < /dev/null & echo started {name} pid $!")
        sys.exit(code)
    elif op == "log":
        n = sys.argv[3] if len(sys.argv) > 3 else "40"
        code, _ = run(c, f"tail -n {n} logs/{sys.argv[2]}.log | tr '\\r' '\\n' | tail -n {n}")
        sys.exit(code)
    elif op == "get":
        sftp = c.open_sftp()
        sftp.get(f"{REMOTE_DIR}/{sys.argv[2]}", sys.argv[3])
        sftp.close()
    else:
        print(__doc__)
        sys.exit(2)


def sh_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
