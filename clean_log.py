"""Merge the concatenated training log(s) into one clean, continuously numbered CSV.

train.py appends to the same CSV when it is resumed, which restarts the ``update``
counter.  This script walks the file, detects every restart and renumbers the
updates continuously, then redraws the training curves from the merged table.
"""

import csv
from pathlib import Path

import numpy as np

import simrl.config as C

SRC = Path("log/ppo_log.csv")
DST = Path("log/ppo_log_merged.csv")

rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
fields = list(rows[0].keys())
offset = 0
prev_raw = 0
prev = 0
for r in rows:
    u = int(r["update"])
    if u <= prev_raw:            # train.py restarted its counter -> new stage
        offset = prev
    prev_raw = u
    prev = offset + u
    r["update"] = prev
print(f"{len(rows)} rows -> updates 1..{prev}")

with DST.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

u = np.array([float(r["update"]) for r in rows])
num = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
ax[0, 0].plot(u, num("return_mean"), color="tab:blue", lw=1)
ax[0, 0].plot(u, num("dist_px_mean") / -10, color="tab:red", lw=1, alpha=0.6)
ax[0, 0].set_title("episode return (blue) and -distance/10 (red)")
ax[0, 1].plot(u, num("success_rate") * 100, color="tab:green", lw=1)
ax[0, 1].set_title("success rate in the rollout (%)")
ax[0, 1].set_ylim(-2, 102)
ax[1, 0].plot(u, num("dist_px_mean"), color="tab:red", lw=1)
ax[1, 0].set_title("mean centre distance during the rollout (px)")
ax[1, 0].axhline(float(C.default_config().episode.success_px), ls=":", color="k", lw=1)
ax[1, 1].plot(u, num("policy_loss"), label="policy", lw=1)
ax[1, 1].plot(u, num("value_loss"), label="value", lw=1)
ax[1, 1].set_yscale("symlog")
ax[1, 1].legend()
ax[1, 1].set_title("losses")
for a in ax.ravel():
    a.set_xlabel("update")
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("log/ppo_curves.png", dpi=130)
print("wrote log/ppo_curves.png")
