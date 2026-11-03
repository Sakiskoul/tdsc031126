import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import rq7_pools as R


def test_control_reduces_eta(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.date_range("2020-01-06", periods=7 * 150, freq="D")
    regime = np.exp(np.cumsum(rng.normal(0, 0.08, len(days))))
    rows = []
    for k, day in enumerate(days):
        dep = rng.poisson(20 * regime[k])
        wit = rng.poisson(20 * regime[max(k - 7, 0)])
        for kind, n in (("deposit", dep), ("withdrawal", wit)):
            rows += [(day + pd.Timedelta(hours=1), 1.0, kind)] * n
    for day in days.strftime("%Y-%m-%d").unique():
        sub = [r for r in rows if r[0].strftime("%Y-%m-%d") == day]
        tbl = pa.table({"ts": pa.array([r[0] for r in sub], pa.timestamp("us")),
                        "denom": pa.array([r[1] for r in sub], pa.float64()),
                        "kind": pa.array([r[2] for r in sub], pa.string())})
        pq.write_table(tbl, tmp_path / f"date={day}.parquet")
    w = R.weekly(tmp_path)
    res = R.analyse_pool(w[w["denom"] == 1.0])
    assert res["weeks"] > 100
    assert res["eta_with_control"] <= res["eta_no_control"] * 1.05
    assert res["covert_srl_eth_week_median"] > 0
