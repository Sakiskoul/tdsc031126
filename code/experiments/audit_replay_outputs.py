from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import data_io as IO


def audit(folder: Path, expected_starts: int | None = None) -> dict:
    files = sorted(folder.glob("start_*.csv")) or sorted(folder.glob("start_*.json"))
    errors, rounding, cases = [], [], 0
    for path in files:
        start = IO.read_checkpoint(path)
        if not np.isfinite(start["check_rel"]) or start["check_rel"] > 1e-7:
            errors.append({"file": path.name, "error": "null-replay mismatch"})
        if not start.get("pool", {}).get("bucket_quota_verified"):
            errors.append({"file": path.name, "error": "pool quota was not certified"})
        with IO.replay_arrays(path.with_suffix(".npz")) as data:

            data = {key: data[key] for key in data.files}
            names = list(map(str, data["names"]))
            draws = list(map(int, data["draws"]))
            laws = list(map(str, data["laws"]))
            intensities = list(map(float, data["intensities"]))
            expected = len(draws) * len(laws) * len(intensities)
            shape = (len(draws), len(laws), len(intensities))
            if (data["scores"].shape != shape + (len(names), 52)
                    or data["stolen"].shape != shape + (52,)
                    or data["keys"].shape != shape
                    or data["base"].shape != (len(names), 52)):
                errors.append({"file": path.name, "error": "unexpected array dimensions"})
                continue
            if len(start["runs"]) != expected:
                errors.append({"file": path.name, "error": "incomplete attack grid"})
            if path.stem != f"start_{start['sid']:03d}":
                errors.append({"file": path.name, "error": "start identifier mismatch"})
            seen = set()
            for ni, name in enumerate(names):
                base = data["base"][ni]
                hit = base >= 1 if name in ("eprocess", "ctm", "threshold") else base > 1
                if int(hit.any()) != start["natural"][name]:
                    issue = {"file": path.name, "detector": name, "error": "natural-alarm curve mismatch"}
                    (rounding if np.isclose(base, 1.0, atol=2e-7, rtol=0).any() else errors).append(issue)
            for run in start["runs"]:
                cases += 1
                pos = (draws.index(run["draw"]), laws.index(run["law"]), intensities.index(run["m"]))
                if pos in seen:
                    errors.append({"file": path.name, "error": "duplicate attack grid cell", "grid": pos})
                seen.add(pos)
                if int(data["keys"][pos]) != run["keys"]:
                    errors.append({"file": path.name, "error": "key count mismatch", "grid": pos})
                if run["keys_week_max"] > 1000 or run["keys"] > 52000:
                    errors.append({"file": path.name, "error": "key budget exceeded", "grid": pos})
                if run["stolen_btc"] > run["planned_btc"] + max(1e-3, 1e-10 * run["planned_btc"]):
                    errors.append({"file": path.name, "error": "value budget exceeded", "grid": pos})
                stored = float(data["stolen"][pos].sum(dtype=np.float64))
                if not np.isfinite(stored) or (data["stolen"][pos] < 0).any():
                    errors.append({"file": path.name, "error": "invalid weekly stolen values", "grid": pos})
                if not np.isclose(stored, run["stolen_btc"], rtol=2e-7, atol=1e-3):
                    errors.append({"file": path.name, "error": "stolen-value array mismatch", "grid": pos})
                for ni, name in enumerate(names):
                    scores = data["scores"][pos + (ni,)]
                    hit = scores >= 1 if name in ("eprocess", "ctm", "threshold") else scores > 1
                    delay = int(hit.argmax()) + 1 if hit.any() else None
                    reported = run["det"][name]["delay"]
                    if bool(run["det"][name]["alarm"]) != (reported is not None):
                        errors.append({"file": path.name, "error": "alarm/delay disagreement", "grid": pos, "detector": name})
                    stop = reported if reported is not None else 52
                    loss = float(data["stolen"][pos][:stop].sum(dtype=np.float64))
                    if not np.isclose(loss, run["det"][name]["loss_btc"], rtol=2e-7, atol=1e-3):
                        errors.append({"file": path.name, "error": "loss-before-alarm mismatch", "grid": pos, "detector": name})
                    if delay != reported:
                        mismatch = {"file": path.name, "grid": pos, "detector": name,
                                    "score_delay": delay, "reported_delay": reported}


                        if np.isclose(scores, 1.0, atol=2e-7, rtol=0).any():
                            rounding.append(mismatch)
                        else:
                            errors.append(mismatch)
            if len(seen) != expected:
                errors.append({"file": path.name, "error": "missing distinct grid cells"})
    ids = {IO.read_checkpoint(p)["sid"] for p in files}
    if expected_starts is not None and ids != set(range(expected_starts)):
        errors.append({"error": "incomplete start set", "missing": sorted(set(range(expected_starts)) - ids)})
    return {"starts": len(files), "expected_starts": expected_starts, "attack_cases": cases, "errors": errors,
            "threshold_rounding_cases": rounding, "passed": bool(files) and not errors and not rounding}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-starts", type=int)
    args = ap.parse_args()
    result = audit(Path(args.runs), args.expected_starts)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
