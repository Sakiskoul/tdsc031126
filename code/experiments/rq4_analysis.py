from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    ev = pd.read_csv(RES / "rq4_puzzle_events.csv", parse_dates=["wk", "t_spend"])
    w = pd.read_csv(RES / f"weekly_{args.tag}.csv", parse_dates=["wk"], index_col="wk")
    ev = ev[(ev["pop"] == "exposed") & (ev["puzzle"] % 5 == 0) & (ev["t_spend"] >= "2019-06-01")]
    g = ev.groupby(["puzzle", "wk"]).agg(btc=("btc", "sum"), t_spend=("t_spend", "min")).reset_index()
    rows = []
    for _, r in g.iterrows():
        if r["wk"] not in w.index:
            continue
        x = w.loc[r["wk"]]
        rows.append({
            "puzzle": int(r["puzzle"]), "week": r["wk"].date().isoformat(), "btc": float(r["btc"]),
            "sigma_btc": float(x["sigma"]), "value_over_sigma": float(r["btc"] / x["sigma"]),
            "z_week": float(x["z_argus"]),
            "alarm_argus": bool(x.get("alarm_argus", False)),
            "alarm_count": bool(x.get("alarm_count", False)),
            "alarm_uncalibrated": bool(x.get("alarm_uncalibrated", False)),
        })
    out = {"events": rows, "n": len(rows),
           "max_value_over_sigma": max((r["value_over_sigma"] for r in rows), default=None),
           "any_alarm_argus": any(r["alarm_argus"] for r in rows)}
    (RES / f"rq4_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
