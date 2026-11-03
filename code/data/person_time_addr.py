from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

VB = """CASE WHEN {v} < 1000000 THEN 'v0_001' WHEN {v} < 10000000 THEN 'v0_01'
           WHEN {v} < 100000000 THEN 'v0_1' WHEN {v} < 1000000000 THEN 'v1'
           WHEN {v} < 10000000000 THEN 'v10' ELSE 'v100' END"""


def ev_where(t_end: str) -> str:
    return f"""t_spend IS NOT NULL AND t_spend >= a AND t_spend <= b AND t_spend < {t_end}
               AND date_trunc('week', a) < date_trunc('week', t_spend)
               AND (pop = 'exposed' OR t_spend = b)"""


def build_bucket(con, seg: str, r_out: Path, e_out: Path, t_end: str) -> None:

    con.execute(f"CREATE OR REPLACE TEMP VIEW s AS SELECT * FROM read_parquet('{seg}')")
    con.execute("""CREATE OR REPLACE TEMP TABLE meta AS
                   SELECT pop, addr, min(era) AS era, any_value(fam) AS fam FROM s GROUP BY pop, addr""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE x AS
        WITH d AS (
          SELECT pop, addr, a AS ts, sat AS dv FROM s
          UNION ALL
          SELECT pop, addr, b AS ts, -sat AS dv FROM s
        ), c AS (
          SELECT pop, addr, ts, CAST(sum(dv) AS BIGINT) AS dv FROM d GROUP BY pop, addr, ts
        )
        SELECT pop, addr, ts, dv,
               CAST(sum(dv) OVER (PARTITION BY pop, addr ORDER BY ts
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS BIGINT) AS v_new
        FROM c""")
    con.execute(f"""
        COPY (
          WITH y AS (SELECT x.*, x.v_new - x.dv AS v_old, m.fam, m.era,
                            CAST(date_trunc('week', x.ts) AS DATE) + 7 AS wk
                     FROM x JOIN meta m USING (pop, addr) WHERE x.dv <> 0)
          SELECT pop, fam, era, vband, wk, sum(dn) AS dn, sum(dv) AS dv, sum(dv2) AS dv2 FROM (
            SELECT pop, fam, era, {VB.format(v='v_old')} AS vband, wk,
                   CAST(-1 AS DOUBLE) AS dn, -CAST(v_old AS DOUBLE) AS dv, -(v_old / 1e8) * (v_old / 1e8) AS dv2
            FROM y WHERE v_old > 0
            UNION ALL
            SELECT pop, fam, era, {VB.format(v='v_new')}, wk,
                   CAST(1 AS DOUBLE), CAST(v_new AS DOUBLE), (v_new / 1e8) * (v_new / 1e8)
            FROM y WHERE v_new > 0
          ) GROUP BY ALL
        ) TO '{r_out.as_posix()}' (FORMAT PARQUET)""")
    con.execute(f"""
        COPY (
          WITH ev AS (
            SELECT pop, addr, CAST(date_trunc('week', t_spend) AS TIMESTAMP) AS wstart
            FROM s WHERE {ev_where(t_end)} GROUP BY pop, addr, wstart
          ), j AS (
            SELECT ev.pop, ev.addr, ev.wstart, x.v_new AS v
            FROM ev ASOF JOIN x ON ev.pop = x.pop AND ev.addr = x.addr AND ev.wstart > x.ts
          )
          SELECT j.pop, m.fam, m.era, {VB.format(v='j.v')} AS vband, CAST(j.wstart AS DATE) AS wk,
                 CAST(count(*) AS DOUBLE) AS n, sum(CAST(j.v AS DOUBLE)) AS sv, sum((j.v / 1e8) * (j.v / 1e8)) AS sv2
          FROM j JOIN meta m ON j.pop = m.pop AND j.addr = m.addr
          WHERE j.v > 0
          GROUP BY ALL
        ) TO '{e_out.as_posix()}' (FORMAT PARQUET)""")


def combine(con, parts: Path, res: Path) -> None:
    pp = parts.as_posix()
    con.execute(f"""
        COPY (
          WITH agg AS (
            SELECT pop, fam, era, vband, wk, sum(dn) AS dn, sum(dv) AS dv, sum(dv2) AS dv2
            FROM read_parquet('{pp}/R_*.parquet') GROUP BY ALL
          )
          SELECT pop, fam, era, vband, 'all' AS band, wk,
                 CAST(sum(dn) OVER w AS DOUBLE) AS n, sum(dv) OVER w AS sv, sum(dv2) OVER w AS sv2
          FROM agg
          WINDOW w AS (PARTITION BY pop, fam, era, vband ORDER BY wk
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        ) TO '{(res / "risk_weekly.parquet").as_posix()}' (FORMAT PARQUET)""")
    con.execute(f"""
        COPY (
          SELECT pop, fam, era, vband, 'all' AS band, wk, CAST(sum(n) AS DOUBLE) AS n, sum(sv) AS sv, sum(sv2) AS sv2,
                 CAST(sum(n) AS DOUBLE) AS n_addr
          FROM read_parquet('{pp}/E_*.parquet') GROUP BY ALL
        ) TO '{(res / "events_weekly.parquet").as_posix()}' (FORMAT PARQUET)""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-end", default="2026-10-01")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--memory", default="10GB")
    ap.add_argument("--work", default="work")
    ap.add_argument("--results", default="results_addr")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    work, res = data / args.work, data / args.results
    parts, done = work / "addr_parts", work / "_done"
    for p in (parts, res, done, data / "tmp"):
        p.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{(data / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    t_end = f"TIMESTAMP '{args.t_end} 00:00:00'"
    for seg in sorted((work / "seg").glob("*.parquet")):
        if seg.name.endswith(".part.parquet"):
            continue
        mark = done / f"ADDR_{seg.stem}"
        if mark.exists():
            continue
        t = time.time()
        build_bucket(con, seg.as_posix(), parts / f"R_{seg.stem}.parquet", parts / f"E_{seg.stem}.parquet", t_end)
        mark.touch()
        print(f"[ADDR_{seg.stem}] {time.time() - t:.0f}s", flush=True)
    combine(con, parts, res)
    big = data / "results" / "big_events.parquet"
    if big.exists():
        (res / "big_events.parquet").write_bytes(big.read_bytes())
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
