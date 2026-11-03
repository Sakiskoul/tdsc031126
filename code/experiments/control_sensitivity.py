from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import data_io as IO

import common as C
import rq


def numeric_panel(results: Path, out: Path) -> pd.DataFrame:
    inputs = out / "numeric_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name in ("risk_weekly", "events_weekly"):
        df = IO.read_frame(results / f"{name}.parquet")


        for col in ("n", "sv", "sv2"):
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        df.to_parquet(inputs / f"{name}.parquet", index=False)
    return C.load_panel(inputs, "2026-10-01")


def remove_recognised_exposure(panel: pd.DataFrame, rows: Path, simple_only: bool = True) -> tuple[pd.DataFrame, dict]:
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    con.execute("SET memory_limit = '2GB'")
    src = (rows / "*.parquet").as_posix().replace("'", "''")
    types = "AND typ IN (2, 3)" if simple_only else ""
    con.execute(f"""CREATE TEMP VIEW intervals AS
        SELECT *, a - INTERVAL 365 DAY AS t_create
        FROM read_parquet('{src}') WHERE hit {types}""")
    con.execute("""CREATE TEMP TABLE segments AS
        WITH x AS (
            SELECT 'control' AS pop, CASE WHEN typ IN (2,3) THEN 'p2pkh' ELSE 'p2sh' END AS fam,
                   CASE WHEN year(t_create) <= 2010 THEN 'e2009_10'
                        WHEN year(t_create) <= 2012 THEN 'e2011_12'
                        WHEN year(t_create) <= 2016 THEN 'e2013_16'
                        WHEN year(t_create) <= 2020 THEN 'e2017_20' ELSE 'e2021_' END AS era,
                   CASE WHEN sat < 1000000 THEN 'v0_001' WHEN sat < 10000000 THEN 'v0_01'
                        WHEN sat < 100000000 THEN 'v0_1' WHEN sat < 1000000000 THEN 'v1'
                        WHEN sat < 10000000000 THEN 'v10' ELSE 'v100' END AS vband,
                   band, sat, b, t_alt, event,
                   greatest(a, t_create + to_days(dlo)) AS sa,
                   least(b, t_create + to_days(dhi)) AS sb
            FROM intervals CROSS JOIN (VALUES ('a1_2',365,730),('a2_5',730,1826),
                ('a5_10',1826,3652),('a10_',3652,40000)) AS bands(band,dlo,dhi)
        ) SELECT * FROM x WHERE sa < sb""")
    weeks = pd.DatetimeIndex(sorted(panel["wk"].unique()))
    con.register("calendar", pd.DataFrame({"wk": weeks}))
    keys = ", ".join(C.KEY)
    risk = con.execute(f"""SELECT {keys}, w.wk, count(*)::DOUBLE AS rm_n,
                 sum(sat / 1e8)::DOUBLE AS rm_sv, sum((sat / 1e8)^2)::DOUBLE AS rm_sv2
        FROM segments s JOIN calendar w
          ON w.wk >= date_trunc('week', greatest(sa,t_alt)) + INTERVAL 7 DAY
         AND w.wk < date_trunc('week', sb) + INTERVAL 7 DAY GROUP BY ALL""").fetchdf()
    events = con.execute(f"""SELECT {keys}, date_trunc('week', b) AS wk,
                 count(*)::DOUBLE AS rm_e_n, sum(sat / 1e8)::DOUBLE AS rm_e_sv,
                 sum((sat / 1e8)^2)::DOUBLE AS rm_e_sv2
        FROM segments WHERE event AND b=sb AND date_trunc('week',sa)<date_trunc('week',b)
        GROUP BY ALL""").fetchdf()
    con.close()
    for df in (risk, events):
        df["wk"] = pd.to_datetime(df["wk"])
    delta = risk.merge(events, on=C.KEY + ["wk"], how="outer").fillna(0)
    delta = delta[delta["wk"].isin(weeks)]
    keep = (panel[(panel["pop"] == "exposed") | (panel["fam"] == "p2pkh")]
            if simple_only else panel).copy()
    check = keep[C.KEY + ["wk"]].merge(delta, on=C.KEY + ["wk"], how="right", indicator=True)
    if (check["_merge"] != "both").any():
        raise ValueError("screening deltas refer to strata absent from the reference panel")
    changed = keep.merge(delta, on=C.KEY + ["wk"], how="left", validate="one_to_one")
    for col in ("n", "sv", "sv2", "e_n", "e_sv", "e_sv2"):
        changed[col] = changed[col] - changed[f"rm_{col}"].fillna(0)
        if changed[col].min() < -1e-6:
            raise ValueError(f"screening would make {col} negative")
        changed[col] = changed[col].clip(lower=0)
    changed = changed[panel.columns]
    old_ex = panel[panel["pop"] == "exposed"].reset_index(drop=True)
    new_ex = changed[changed["pop"] == "exposed"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_ex, new_ex)
    info = {"kept_control_family": "p2pkh (P2PKH/P2WPKH)" if simple_only else "all original families",
            "exposed_panel_unchanged": True,
            "known_reveal_events_removed": float(events["rm_e_n"].sum()),
            "known_reveal_event_btc_removed": float(events["rm_e_sv"].sum()),
            "removed_risk_output_weeks": float(risk["rm_n"].sum())}
    return changed, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-families", action="store_true", help="isolate recognised reveals, retaining all original script families")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = numeric_panel(Path(args.results), out)
    screened, info = remove_recognised_exposure(panel, Path(args.rows), simple_only=not args.all_families)
    screened.to_parquet(out / "screened_panel.parquet", index=False)
    print("screened", info, flush=True)
    setup = rq.Setup(screened, ("2014-01-06", "2019-12-30"), ("2020-01-06", "2026-09-28"),
                     target_arl=104.0, refit=13)
    result = {"cohort": info, "rq2": rq.rq2_control(setup), "rq3": rq.rq3_false_alarms(setup),
              "seconds": time.time() - t0,
              "scope": "historical output-panel sensitivity; recognised cross-format reveals removed; residual script and x-only exposures remain unaudited"}
    (out / "sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
