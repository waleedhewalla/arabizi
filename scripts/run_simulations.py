"""Re-run every design-sensitivity scenario and write the results.

Usage:
    python scripts/run_simulations.py [--seed 20260926] [--reps N] [--only name ...]

Each scenario gets its own child generator spawned from the master seed, so a
single scenario can be re-run alone and still reproduce its published numbers.
Output: results/<scenario>.csv, results/SUMMARY.md, results/environment.txt
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arabizi_toolkit import __version__  # noqa: E402
from arabizi_toolkit.simulate import SCENARIOS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--reps", type=int, default=None, help="override repetitions for every scenario")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    names = list(SCENARIOS)
    children = np.random.SeedSequence(args.seed).spawn(len(names))
    summary = [f"# Simulation results\n\nseed `{args.seed}` · toolkit {__version__} · "
               f"numpy {np.__version__} · scipy {scipy.__version__} · python {platform.python_version()}\n"]
    for name, ss in zip(names, children):
        if args.only and name not in args.only:
            continue
        rng = np.random.default_rng(ss)
        kwargs = {"reps": args.reps} if args.reps else {}
        t0 = time.time()
        rows = SCENARIOS[name](rng, **kwargs)
        dt = time.time() - t0
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        head = list(rows[0])
        summary.append(f"\n## {name}\n\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for r in rows:
            summary.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v)
                                              for v in r.values()) + " |")
        print(f"{name}: {len(rows)} rows in {dt:.0f}s", flush=True)
    if not args.only:
        (OUT / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "environment.txt").write_text(
        f"python {platform.python_version()}\nnumpy {np.__version__}\nscipy {scipy.__version__}\n"
        f"toolkit {__version__}\nseed {args.seed}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
