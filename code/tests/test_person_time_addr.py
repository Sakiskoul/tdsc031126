import sys
from pathlib import Path

import duckdb
import pandas as pd
import data_io as IO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import person_time_addr as PA


def test_address_value_runs_and_event_carries_whole_unit(tmp_path):
    W = lambda k: pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=k)
    mid = pd.Timedelta(days=2)


    seg = pd.DataFrame({
        "pop": ["exposed", "exposed"], "addr": [b"A", b"A"], "fam": ["p2pkh", "p2pkh"],
        "era": ["e2013_16", "e2017_20"], "vband": ["v1", "v1"], "sat": [100_000_000, 200_000_000],
        "t_spend": [W(5) + mid, pd.NaT], "band": ["a2_5", "a2_5"],
        "a": [W(1) + mid, W(3) + mid], "b": [W(5) + mid, pd.Timestamp("2026-10-01")],
    })
    sp = tmp_path / "seg.parquet"
    seg.to_parquet(sp)
    con = duckdb.connect()
    PA.build_bucket(con, sp.as_posix(), tmp_path / "R_0.parquet", tmp_path / "E_0.parquet",
                    "TIMESTAMP '2026-10-01 00:00:00'")
    PA.combine(con, tmp_path, tmp_path)
    risk = IO.read_frame(tmp_path / "risk_weekly.parquet").sort_values("wk")
    ev = IO.read_frame(tmp_path / "events_weekly.parquet")

    def at(k):
        r = risk[pd.to_datetime(risk["wk"]) <= W(k)]
        return r.groupby("vband").tail(1).query("n > 0")

    assert at(2)[["n", "sv"]].values.tolist() == [[1.0, 1e8]]
    assert at(4)[["n", "sv"]].values.tolist() == [[1.0, 3e8]]
    assert at(6)[["n", "sv"]].values.tolist() == [[1.0, 2e8]]
    assert (risk["era"] == "e2013_16").all()
    assert len(ev) == 1 and ev["n"].iloc[0] == 1.0
    assert ev["sv"].iloc[0] == 3e8
    assert pd.Timestamp(ev["wk"].iloc[0]) == W(5)
