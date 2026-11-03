from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import replay as R


Light = R.Light


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    light, *_ = pd.read_pickle(args.cache)
    results = []
    for sid, t0 in enumerate(R.starts(light.panel)):
        replay = R.Replayer(light, t0)
        weekly, _, _ = replay.stats(light.panel, band=False)
        ref = light.w_cal.reindex(replay.win)
        errors = {}
        for col in ("S", "var"):
            scale = max(float(ref[col].abs().median()), 1e-12)
            errors[col] = float((weekly[col] - ref[col]).abs().max() / scale)
        row = {"sid": sid, "t0": str(t0.date()), "relative_errors": errors,
               "passed": all(np.isfinite(v) and v <= 1e-7 for v in errors.values())}
        results.append(row)
        print(sid, row["t0"], errors, flush=True)
    out = {"starts": len(results), "all_passed": all(r["passed"] for r in results), "checks": results}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("all_passed", out["all_passed"], flush=True)
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
