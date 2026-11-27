"""Compare the smoothness of two rollouts recorded at the 2 ms physics rate.

Both inputs are the ``*_substeps.npz`` files written by ``evaluate.py``.  The
interesting question is not the sampled value itself but how the *derivative*
behaves at the 40 ms control-period boundaries (index % 20 == 0) compared with
inside a period.  A zero-order-hold position target produces a visible jump in
joint acceleration / torque at every boundary; a smooth reference does not.

Usage:
    python analyze_smoothness.py log/smoke_step_substeps.npz \
                                 log/eval_filt_substeps.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NSUB = 20  # 20 x 2 ms control period


def _metrics(path: str) -> dict:
    z = np.load(path)
    t = np.asarray(z["t"], dtype=np.float64)
    qacc = np.asarray(z["qacc7"], dtype=np.float64)
    tau = np.asarray(z["tau"], dtype=np.float64)
    tip = np.asarray(z["fingertip"], dtype=np.float64)
    dt = float(np.median(np.diff(t)))

    dqacc = np.diff(qacc, axis=0)
    dtau = np.diff(tau, axis=0)
    # diff i is the step from sample i to i+1.  Sample 0 is t=0, samples
    # 1..20 are the first control period, so i=20,40,... cross a boundary.
    boundary = np.arange(len(dqacc)) % NSUB == 0
    inside = ~boundary

    vel = np.gradient(tip, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.gradient(vel, dt, axis=0)
    accn = np.linalg.norm(acc, axis=1)

    def ratio(a, b):
        return float(np.mean(a) / max(np.mean(b), 1e-12))

    return {
        "file": path,
        "duration_s": float(t[-1]),
        "samples": int(len(t)),
        "boundary_dqacc": float(np.mean(np.abs(dqacc[boundary]))),
        "inside_dqacc": float(np.mean(np.abs(dqacc[inside]))),
        "boundary_dqacc_ratio": ratio(np.abs(dqacc[boundary]), np.abs(dqacc[inside])),
        "boundary_dtau": float(np.mean(np.abs(dtau[boundary]))),
        "inside_dtau": float(np.mean(np.abs(dtau[inside]))),
        "boundary_dtau_ratio": ratio(np.abs(dtau[boundary]), np.abs(dtau[inside])),
        "speed_max": float(speed.max()),
        "speed_mean": float(speed.mean()),
        "acc_max": float(accn.max()),
        "acc_mean": float(accn.mean()),
        "acc_p95": float(np.percentile(accn, 95)),
        "jerk_max": float(np.max(np.linalg.norm(np.gradient(acc, dt, axis=0), axis=1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="candidate")
    ap.add_argument("--out", default="log/smoothness_compare.json")
    args = ap.parse_args()

    a = _metrics(args.baseline)
    b = _metrics(args.candidate)
    out = {args.label_a: a, args.label_b: b}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    keys = [
        "duration_s",
        "samples",
        "boundary_dqacc",
        "inside_dqacc",
        "boundary_dqacc_ratio",
        "boundary_dtau",
        "inside_dtau",
        "boundary_dtau_ratio",
        "speed_max",
        "speed_mean",
        "acc_max",
        "acc_mean",
        "acc_p95",
        "jerk_max",
    ]
    print(f"{'metric':26s} {args.label_a:>14s} {args.label_b:>14s}  change")
    for k in keys:
        va, vb = a[k], b[k]
        if isinstance(va, int):
            print(f"{k:26s} {va:14d} {vb:14d}")
        else:
            change = (vb / va - 1.0) * 100.0 if va else float("nan")
            print(f"{k:26s} {va:14.4f} {vb:14.4f}  {change:+7.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
