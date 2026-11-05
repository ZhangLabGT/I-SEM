#!/usr/bin/env python3
"""K562 supervisor. Owns every K562 fit left, releases claims of dead owners, reruns crashed fits (identical settings)
one per process, records every crash, and stops when each expected fit has output or is FAILED. Starts a K562 fit only when
free memory covers it; that changes when it runs, not its settings. python3 -u a1.py"""
import csv, glob, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRASH = HERE / "results/crashes.csv"
ARMS = ("no_half", "no_push", "no_push_signed", "no_se", "no_stability", "shipped", "wald")
SCREENS = ("hepg2", "rpe1", "jurkat", "k562")
NEED_MIB = {"hepg2": 3000, "rpe1": 8000, "jurkat": 14000, "k562": 34000, "sims": 3000}   # free memory needed to start one
K562_SETTLE, MAX_SLOTS, MAX_CRASHES = 180, 10, 3                                      # first run + two reruns


def gpu():
    u, t = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    return int(u), int(t)


def record(arm, ds, when, used, how):
    new = not CRASH.exists()
    with CRASH.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "dataset", "time", "gpu_used_mib", "gpu_total_mib", "how"])
        w.writerow([arm, ds, when, used, 143771, how]); f.flush()
    print(f"CRASH {arm} {ds} {when} {used} {how}", flush=True)


def crashes():
    n = {}
    if CRASH.exists():
        for r in csv.DictReader(CRASH.open()):
            n[r["arm"], r["dataset"]] = n.get((r["arm"], r["dataset"]), 0) + 1
    return n


def expected():
    """(kind, dataset key, harness name, arm): weak + contrast from the block lists written by this script's first call."""
    out = []
    for blk in ("weak", "contrast"):
        for line in (HERE / f"tmp/block_{blk}.txt").read_text().split():
            key = "/".join(Path(line).parts[-2:])
            out += [("sims", key, line, a) for a in ARMS]
    return out + [("real", ds, ds, a) for ds in SCREENS for a in ARMS]


def paths(kind, key, arm):
    base = HERE / "cache" / kind / key / {"no_push_abs": "no_push"}.get(arm, arm)
    return base.with_suffix(".npz"), base.with_suffix(".claim"), base.with_suffix(".err")


def externals():
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", "harness.py fit"], capture_output=True, text=True).stdout.split()
    return [p for p in r if p != str(os.getpid())]


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stale_claims():
    """A claim with a pid is dead if that pid is gone. An old empty claim has no owner on record, so it is stale only when no
    harness that could own it is left. Live owners' claims are never touched."""
    ext = externals()
    for f in glob.glob(str(HERE / "cache/**/*.claim"), recursive=True):
        try:
            txt = Path(f).read_text().strip()
            if txt.startswith("a1:"):
                continue
            if (txt and not alive(int(txt))) or (not txt and not ext):
                rel = Path(f).relative_to(HERE / "cache").with_suffix("")
                key = "/".join(rel.parts[1:-1]); used, _ = gpu()
                if not Path(f).with_suffix(".npz").exists():
                    record(rel.name, key, time.strftime("%T"), used, "found: claim of a dead worker, no output")
                os.unlink(f)
        except FileNotFoundError:
            pass


def claim(c):
    c.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"a1:{os.getpid()}".encode()); os.close(fd)
    return True


def main():
    for blk in ("weak", "contrast"):
        if not (HERE / f"tmp/block_{blk}.txt").exists():
            d = subprocess.run(["bash", "run.sh", "-c", f"import harness as h; print(' '.join(str(x) for x in h.block('{blk}')))"], capture_output=True, text=True, cwd=HERE)
            (HERE / f"tmp/block_{blk}.txt").write_text(d.stdout.strip().splitlines()[-1])
    exp = expected()
    assert sum(x[0] == "sims" for x in exp) == 2240 and sum(x[0] == "real" for x in exp) == 28
    for a in ("wald", "no_push_signed", "no_push", "no_stability", "no_half"):          # K562 stays with this gated supervisor
        npz, c, _ = paths("real", "k562", a)
        if not npz.exists():
            claim(c)
    kids, held_k562, last_k562 = {}, set(), 0.
    while True:
        stale_claims()
        for t, (p, fit, log) in list(kids.items()):
            if p.poll() is None:
                continue
            kind, key, name, arm = fit
            npz, c, err = paths(kind, key, arm)
            used, _ = gpu()
            if not npz.exists() and not err.exists():
                record(arm, key, time.strftime("%T"), used, f"rerun aborted, rc={p.returncode}")
            c.unlink(missing_ok=True)
            print(f"{time.strftime('%T')} end {arm} {key} rc={p.returncode} {'ok' if npz.exists() else 'NO OUTPUT'} free {143771 - used}", flush=True)
            del kids[t]
        n = crashes(); ext = externals()
        todo = []
        for fit in exp:
            kind, key, name, arm = fit
            npz, c, err = paths(kind, key, arm)
            if npz.exists() or err.exists() or n.get((arm, key), 0) >= MAX_CRASHES or any(fit == v[1] for v in kids.values()):
                continue
            mine = c.exists() and c.read_text().startswith("a1:")
            if c.exists() and not mine:
                continue
            if ext and not mine and n.get((arm, key), 0) == 0:   # still in a live harness's queue
                continue
            todo.append((fit, mine))
        if not todo and not kids and not ext:
            break
        order = sorted(todo, key=lambda x: x[0][1] != "k562")     # K562 first while it has room
        for fit, mine in order:
            kind, key, name, arm = fit
            used, total = gpu(); free = total - used
            gate = "k562" if key == "k562" else (key if kind == "real" else "sims")
            if len(kids) >= MAX_SLOTS or free < NEED_MIB[gate]:
                continue
            if key == "k562" and time.time() - last_k562 < K562_SETTLE:
                continue
            _, c, _ = paths(kind, key, arm)
            if not mine and not claim(c):
                continue
            log = open(HERE / f"logs/a1_{arm}_{key.replace('/', '_')}_{int(time.time())}.log", "w")
            p = subprocess.Popen(["bash", "run.sh", "harness.py", "one", "--kind", kind, "--name", name, "--arms", arm], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            kids[time.time()] = (p, fit, log)
            if key == "k562":
                last_k562 = time.time()
            print(f"{time.strftime('%T')} start {arm} {key} free {free} (crashes so far {n.get((arm, key), 0)})", flush=True)
            time.sleep(2)
        time.sleep(30)
    print("a1 done", flush=True)


if __name__ == "__main__":
    main()
