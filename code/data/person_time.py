from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

FAM = """CASE typ WHEN 1 THEN 'p2pk' WHEN 6 THEN 'p2tr' WHEN 2 THEN 'p2pkh' WHEN 3 THEN 'p2pkh'
         ELSE 'p2sh' END"""
ERA = """CASE WHEN year(t_create) <= 2010 THEN 'e2009_10' WHEN year(t_create) <= 2012 THEN 'e2011_12'
         WHEN year(t_create) <= 2016 THEN 'e2013_16' WHEN year(t_create) <= 2020 THEN 'e2017_20'
         ELSE 'e2021_' END"""
VBAND = """CASE WHEN sat < 1000000 THEN 'v0_001' WHEN sat < 10000000 THEN 'v0_01'
           WHEN sat < 100000000 THEN 'v0_1' WHEN sat < 1000000000 THEN 'v1'
           WHEN sat < 10000000000 THEN 'v10' ELSE 'v100' END"""


def step(con, name: str, out: Path, sql: str) -> None:
    if out.exists():
        print(f"[{name}] checkpoint exists, skipped")
        return
    t = time.time()
    tmp = out.with_suffix(".part.parquet")
    con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.rename(out)
    n = con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    print(f"[{name}] {n:,} rows in {time.time() - t:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-end", default="2026-10-01")
    ap.add_argument("--d-min-days", type=int, default=365)
    ap.add_argument("--dust-sat", type=int, default=100_000)
    ap.add_argument("--big-btc", type=int, default=10)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--memory", default="40GB")
    ap.add_argument("--files-until", default=None, help="only read daily files up to this date")
    ap.add_argument("--work", default="work", help="checkpoint folder name")
    ap.add_argument("--results", default="results", help="results folder name")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    work = data / args.work
    res = data / args.results
    for p in (work, res, data / "tmp"):
        p.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{(data / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    def files(kind: str) -> str:
        fs = sorted(p for p in (data / kind).glob("date=*.parquet")
                    if args.files_until is None or p.stem.split("=")[1] <= args.files_until)
        return "[" + ", ".join("'" + p.as_posix() + "'" for p in fs) + "]"

    outs = files("outputs")
    ins = files("inputs")
    t_end = f"TIMESTAMP '{args.t_end} 00:00:00'"

    life = work / "life.parquet"
    step(con, "A life", life, f"""
        SELECT o.addr, o.typ, o.sat, o.ts AS t_create, i.ts AS t_spend
        FROM read_parquet({outs}) AS o
        LEFT JOIN read_parquet({ins}) AS i ON i.ptx = o.tx AND i.pvout = o.vout
        WHERE o.ts < {t_end}
    """)

    reveal = work / "reveal.parquet"
    step(con, "B reveal", reveal, f"""
        SELECT addr, min(CASE WHEN typ = 1 THEN t_create ELSE t_spend END) AS t_rev
        FROM '{life.as_posix()}'
        WHERE addr IS NOT NULL
        GROUP BY addr
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW iv AS
        SELECT l.addr, {FAM} AS fam, l.sat, l.t_create,
               CASE WHEN l.t_spend < {t_end} THEN l.t_spend END AS t_spend,
               l.t_create + INTERVAL ({args.d_min_days}) DAY AS t_dorm,
               CASE WHEN typ IN (1, 6) THEN l.t_create ELSE r.t_rev END AS t_exp,
               r.t_rev,
               {ERA} AS era, {VBAND} AS vband
        FROM '{life.as_posix()}' AS l LEFT JOIN '{reveal.as_posix()}' AS r USING (addr)
        WHERE l.sat >= {args.dust_sat}
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ivals AS
        SELECT 'exposed' AS pop, addr, fam, era, vband, sat, t_create, t_spend,
               greatest(t_exp, t_dorm) AS a, coalesce(t_spend, {t_end}) AS b
        FROM iv WHERE t_exp IS NOT NULL AND t_exp < {t_end}
        UNION ALL
        SELECT 'control', addr, fam, era, vband, sat, t_create, t_spend,
               t_dorm, least(coalesce(t_spend, {t_end}), coalesce(t_rev, {t_end}))
        FROM iv WHERE fam IN ('p2pkh', 'p2sh')
    """)


    con.execute("""
        CREATE OR REPLACE TEMP TABLE bands AS
        SELECT * FROM (VALUES ('a1_2', 365, 730), ('a2_5', 730, 1826), ('a5_10', 1826, 3652),
                              ('a10_', 3652, 40000)) AS t(band, d_lo, d_hi)
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW seg AS
        SELECT pop, addr, fam, era, vband, sat, t_spend, band,
               greatest(a, t_create + to_days(d_lo)) AS a,
               least(b, t_create + to_days(d_hi)) AS b
        FROM ivals CROSS JOIN bands
        WHERE a < b
          AND greatest(a, t_create + to_days(d_lo)) < least(b, t_create + to_days(d_hi))
    """)

    segp = work / "seg.parquet"
    step(con, "C seg", segp, "SELECT * FROM seg")

    risk = res / "risk_weekly.parquet"
    step(con, "D risk", risk, f"""
        WITH deltas AS (
          SELECT pop, fam, era, vband, band, CAST(date_trunc('week', a) AS DATE) + 7 AS wk,
                 1 AS dn, sat AS dv, (sat / 1e8) * (sat / 1e8) AS dv2
          FROM '{segp.as_posix()}'
          UNION ALL
          SELECT pop, fam, era, vband, band, CAST(date_trunc('week', b) AS DATE) + 7,
                 -1, -sat, -(sat / 1e8) * (sat / 1e8)
          FROM '{segp.as_posix()}'
        ), agg AS (
          SELECT pop, fam, era, vband, band, wk, sum(dn) AS dn, sum(dv) AS dv, sum(dv2) AS dv2
          FROM deltas GROUP BY ALL
        )
        SELECT pop, fam, era, vband, band, wk,
               sum(dn) OVER w AS n, sum(dv) OVER w AS sv, sum(dv2) OVER w AS sv2
        FROM agg
        WINDOW w AS (PARTITION BY pop, fam, era, vband, band ORDER BY wk
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """)

    events = res / "events_weekly.parquet"
    ev_where = f"""t_spend IS NOT NULL AND t_spend >= a AND t_spend <= b AND t_spend < {t_end}
                   AND date_trunc('week', a) < date_trunc('week', t_spend)
                   AND (pop = 'exposed' OR t_spend = b)"""
    step(con, "E events", events, f"""
        SELECT pop, fam, era, vband, band, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
               count(*) AS n, sum(sat) AS sv, sum((sat / 1e8) * (sat / 1e8)) AS sv2,
               count(DISTINCT addr) AS n_addr
        FROM '{segp.as_posix()}' WHERE {ev_where}
        GROUP BY ALL
    """)


    wake = res / "wakeups_weekly.parquet"
    step(con, "F wakeups", wake, f"""
        WITH w AS (
          SELECT pop, addr, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
                 sum(sat) AS v, count(*) AS n_out, min(era) AS era, any_value(fam) AS fam
          FROM '{segp.as_posix()}' WHERE {ev_where}
          GROUP BY pop, addr, wk
        )
        SELECT pop, fam, era, wk, count(*) AS n_units, sum(v) AS sv,
               sum((v / 1e8) * (v / 1e8)) AS sv2, max(v) AS vmax
        FROM w GROUP BY ALL
    """)

    big = res / "big_events.parquet"
    step(con, "G big", big, f"""
        SELECT pop, addr, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
               sum(sat) AS sat, count(*) AS n_out, min(era) AS era, any_value(fam) AS fam,
               max(band) AS band
        FROM '{segp.as_posix()}' WHERE {ev_where} AND pop = 'exposed'
        GROUP BY pop, addr, wk
        HAVING sum(sat) >= {args.big_btc} * 100000000
    """)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
