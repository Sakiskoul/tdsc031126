from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    panel = C.load_panel(Path(args.results), "2026-10-01")
    weeks = sorted(panel["wk"].unique())
    last = weeks[-1]
    hist = panel[panel["wk"] >= weeks[-104]]
    t = time.perf_counter()
    model = C.NullModel().fit(hist)
    t_refit = time.perf_counter() - t
    week = panel[panel["wk"] == last]

    recent = panel[panel["wk"] >= weeks[-16]]
    det = C.RobustDetector()
    edet = C.EDetector(c=104.0)
    times = []
    for _ in range(args.reps):
        t = time.perf_counter()
        w = C.weekly_stats(recent, model)
        z = det.z(w.iloc[[-1]])
        det.run(np.asarray(z))
        edet.run(edet.y(w.iloc[[-1]]))
        times.append(time.perf_counter() - t)
    out = {"strata_rows_per_week": int(len(week)), "refit_seconds": t_refit,
           "weekly_cycle_ms_median": 1000 * float(np.median(times)),
           "weekly_cycle_ms_p90": 1000 * float(np.quantile(times, 0.9))}
    C.save_json("bench_detection", out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
