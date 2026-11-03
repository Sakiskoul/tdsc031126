from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 0.1


def weekly(folder: Path) -> pd.DataFrame:
    if folder.is_file() and folder.suffix == '.csv':
        return pd.read_csv(folder,parse_dates=['wk'],float_precision='round_trip')
    q = f"""
      SELECT denom, date_trunc('week', ts) AS wk,
             sum(CASE WHEN kind = 'deposit' THEN 1 ELSE 0 END) AS D,
             sum(CASE WHEN kind = 'withdrawal' THEN 1 ELSE 0 END) AS W
      FROM read_parquet('{(folder / 'date=*.parquet').as_posix()}')
      GROUP BY ALL ORDER BY denom, wk
    """
    df = duckdb.connect().execute(q).fetchdf()
    df["wk"] = pd.to_datetime(df["wk"])
    out = []
    for d, g in df.groupby("denom"):
        idx = pd.date_range(g["wk"].min(), g["wk"].max(), freq="W-MON")
        g = g.set_index("wk").reindex(idx).fillna({"D": 0, "W": 0})
        g["denom"] = d
        g["B"] = d * (g["D"].cumsum() - g["W"].cumsum())
        out.append(g.rename_axis("wk").reset_index())
    return pd.concat(out, ignore_index=True)


def analyse_pool(g: pd.DataFrame, lag: int = 4) -> dict:
    g = g.copy().set_index("wk")
    d = float(g["denom"].iloc[0])

    g["lam_nc"] = g["W"].shift(1).rolling(lag, min_periods=lag).median()

    g["ldep"] = np.log(g["D"].shift(1).rolling(lag, min_periods=lag).mean().clip(lower=0.5))
    g["lwit"] = np.log(g["W"].shift(1).rolling(lag, min_periods=lag).mean().clip(lower=0.5))
    fit = g.dropna(subset=["ldep", "lwit"])
    X = sm.add_constant(fit[["ldep", "lwit"]].values)
    glm = sm.GLM(fit["W"].values, X, family=sm.families.Poisson()).fit()
    g.loc[fit.index, "lam_c"] = glm.predict(X)
    ok = g.dropna(subset=["lam_nc", "lam_c"])
    ok = ok[(ok["lam_nc"] > 0) & (ok["W"] > 0)]
    eta_nc = float(np.median(np.abs(np.log(ok["W"] / ok["lam_nc"]))))
    eta_c = float(np.median(np.abs(np.log(ok["W"] / ok["lam_c"]))))
    lam = ok["lam_c"]
    srl = 2 * EPS * d * np.sqrt(lam)
    lin_nc = eta_nc * d * lam
    lin_c = eta_c * d * lam
    B = ok["B"].clip(lower=0)
    return {
        "denom_eth": d, "weeks": int(len(ok)),
        "withdrawals_per_week_median": float(ok["W"].median()),
        "float_eth_median": float(B.median()),
        "eta_no_control": eta_nc, "eta_with_control": eta_c,
        "glm_coef_log_deposits": float(glm.params[1]), "glm_coef_log_withdrawals": float(glm.params[2]),
        "covert_srl_eth_week_median": float(srl.median()),
        "covert_linear_eth_week_median_no_control": float(lin_nc.median()),
        "covert_linear_eth_week_median_with_control": float(lin_c.median()),
        "weeks_to_drain_float_srl_median": float((B / srl.clip(lower=1e-9)).median()),
        "weeks_to_drain_float_linear_nc_median": float((B / lin_nc.clip(lower=1e-9)).median()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tornado", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--split", default="2022-08-08", help="sanctions date, analysed separately")
    args = ap.parse_args()
    w = weekly(Path(args.tornado))
    out = {"eps": EPS, "pools": {}}
    for period, sel in (("all", slice(None)), ("pre_sanctions", slice(None, args.split)),
                        ("post_sanctions", slice(args.split, None))):
        res = {}
        for d, g in w.groupby("denom"):
            gg = g.set_index("wk").loc[sel].reset_index()
            if len(gg) > 20:
                res[str(d)] = analyse_pool(gg)
        out["pools"][period] = res
    root = Path(__file__).resolve().parents[2]
    (root / "results").mkdir(exist_ok=True)
    path = root / "results" / f"rq7_pools_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    w.to_csv(root / "results" / f"rq7_weekly_{args.tag}.csv", index=False)
    print(json.dumps(out["pools"]["all"], indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
