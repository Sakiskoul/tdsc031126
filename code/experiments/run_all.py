from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import data_io as IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import rq
import rq56


def export_weekly(s, tag: str) -> None:

    import numpy as np
    w = s.w_cal.copy()
    w["E_eff"] = s.phi * w["energy"]
    w["sigma"] = np.sqrt(s.phi * w["var"])
    for name, spec in rq.detectors(s).items():
        col, var_col, calibrated, det = spec
        ww = (s.w_cal if calibrated else s.w_unc)

        w[f"alarm_{name}"] = rq.run_detector(s, name, spec).reindex(w.index).fillna(False).astype(bool).values
        if col == "E":
            w["y_eprocess"] = det.y(ww)
        elif col in ("S", "Sn"):
            if col == "Sn":
                ww = rq._count_stats(ww)
            w[f"z_{name}"] = det.z(ww, col, var_col)
    th = C.control_theta(s.panel, s.model)
    w = w.join(th[["theta", "theta_raw"]], how="left")
    w = w.join(C.exposed_theta(s.panel, s.model).rename("theta_exp"), how="left")
    w = w.join(C.drift_series(s.panel, s.model), how="left")
    w["drift_margin"] = C.drift_margin(s.panel, s.model, s.train)
    w["split"] = np.where(w.index < pd.Timestamp(s.evalw[0]), "train", "eval")
    w.to_csv(C.RESULTS / f"weekly_{tag}.csv", index_label="wk")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--t-end", required=True)
    ap.add_argument("--train", nargs=2, required=True)
    ap.add_argument("--eval", nargs=2, required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--arl", type=float, default=104.0)
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--refit", type=int, default=13)
    args = ap.parse_args()

    t = time.time()
    res = Path(args.results)
    panel = C.load_panel(res, args.t_end)
    big = IO.read_frame(res / "big_events.parquet")
    s = rq.Setup(panel, tuple(args.train), tuple(args.eval), target_arl=args.arl, refit=args.refit)
    print(f"panel {panel.shape}, phi {s.phi:.2f}, {time.time() - t:.0f}s", flush=True)

    out = {}
    out["rq1"] = rq.rq1_budget(s)
    print("RQ1", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["rq1"].items()}, flush=True)
    out["rq2"] = rq.rq2_control(s)
    print("RQ2", out["rq2"], flush=True)
    out["rq3"] = rq.rq3_false_alarms(s, big)
    print("RQ3", {k: (v["alarms"] if isinstance(v, dict) and "alarms" in v else v) for k, v in out["rq3"].items()}, flush=True)
    out["rq5"] = rq56.rq5_sweep(s, big, horizon=args.horizon)
    print("RQ5 natural alarms", out["rq5"]["natural_alarms"], "sigma", round(out["rq5"]["sigma_week_median_btc"], 1),
          "theory", round(out["rq5"]["theory_covert_btc_eps0.1"], 1), flush=True)
    for law, lres in out["rq5"]["laws"].items():
        print("RQ5", law, {d: [round(x, 2) for x in r["detection_rate"]] for d, r in lres.items()}, flush=True)
    out["rq6"] = rq56.rq6_response(s)
    export_weekly(s, args.tag)
    out["meta"] = {"results": res.name, "t_end": args.t_end, "train": args.train, "eval": args.eval,
                   "arl": args.arl, "horizon": args.horizon, "eps": rq.EPS, "seconds": time.time() - t}
    path = C.save_json(f"experiments_{args.tag}", out)
    print("saved", path, f"in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
