from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import duckdb

from person_time import ERA, FAM, VBAND


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-end", default="2026-10-01")
    ap.add_argument("--d-min-days", type=int, default=365)
    ap.add_argument("--dust-sat", type=int, default=100_000)
    ap.add_argument("--big-btc", type=int, default=10)
    ap.add_argument("--parts", type=int, default=64)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--memory", default="14GB")
    ap.add_argument("--files-until", default=None)
    ap.add_argument("--work", default="work")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    work, res, tmpd = data / args.work, data / args.results, data / "tmp"
    done = work / "_done"
    for p in (work, res, tmpd, done):
        p.mkdir(parents=True, exist_ok=True)
    K = args.parts
    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{tmpd.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    t_end = f"TIMESTAMP '{args.t_end} 00:00:00'"

    def files(kind: str) -> str:
        fs = sorted(p for p in (data / kind).glob("date=*.parquet")
                    if args.files_until is None or p.stem.split("=")[1] <= args.files_until)
        return "[" + ", ".join("'" + p.as_posix() + "'" for p in fs) + "]"

    def unit(name: str, fn) -> None:
        mark = done / name
        if mark.exists():
            return
        t = time.time()
        fn()
        mark.touch()
        print(f"[{name}] {time.time() - t:.0f}s", flush=True)

    def copy(sql: str, out: Path) -> None:
        tmp = out.with_suffix(".part.parquet")
        con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        tmp.rename(out)


    def bucket(kind: str, cols: str, key: str, where: str) -> None:
        d = work / f"{kind}_by_tx"
        shutil.rmtree(d, ignore_errors=True)
        con.execute(f"""COPY (SELECT {cols}, CAST(hash({key}) % {K} AS INTEGER) AS p
                            FROM read_parquet({files(kind)}) {where})
                        TO '{d.as_posix()}' (FORMAT PARQUET, PARTITION_BY (p), COMPRESSION ZSTD)""")
    unit("P_outputs", lambda: bucket("outputs", "tx, vout, ts, typ, addr, sat", "tx", f"WHERE ts < {t_end}"))
    unit("P_inputs", lambda: bucket("inputs", "ptx, pvout, ts", "ptx", ""))


    life = work / "life"
    for k in range(K):
        def join_k(k=k):
            for f in life.glob(f"*/tx{k}_*.parquet"):
                f.unlink()
            o = (work / "outputs_by_tx" / f"p={k}").as_posix()
            i = (work / "inputs_by_tx" / f"p={k}").as_posix()
            con.execute(f"""COPY (SELECT o.addr, o.typ, o.sat, o.ts AS t_create, i.ts AS t_spend,
                                         CAST(hash(o.addr) % {K} AS INTEGER) AS ap
                                  FROM read_parquet('{o}/*.parquet') AS o
                                  LEFT JOIN read_parquet('{i}/*.parquet') AS i
                                         ON i.ptx = o.tx AND i.pvout = o.vout)
                            TO '{life.as_posix()}' (FORMAT PARQUET, PARTITION_BY (ap),
                               FILENAME_PATTERN 'tx{k}_{{i}}', OVERWRITE_OR_IGNORE true,
                               COMPRESSION ZSTD)""")
        if (work / "inputs_by_tx" / f"p={k}").exists() or (work / "outputs_by_tx" / f"p={k}").exists():
            unit(f"A_{k}", join_k)


    seg_dir, rev_dir, part_dir = work / "seg", work / "reveal", work / "parts"
    for p in (seg_dir, rev_dir, part_dir):
        p.mkdir(exist_ok=True)
    buckets = sorted(d for d in life.iterdir() if d.is_dir())
    bands = """(VALUES ('a1_2', 365, 730), ('a2_5', 730, 1826), ('a5_10', 1826, 3652),
                       ('a10_', 3652, 40000)) AS bands(band, d_lo, d_hi)"""
    for d in buckets:
        j = d.name.split("=")[1]
        lj = f"'{d.as_posix()}/*.parquet'"
        rj, sj = rev_dir / f"{j}.parquet", seg_dir / f"{j}.parquet"

        def bc(lj=lj, rj=rj, sj=sj):
            copy(f"""SELECT addr, min(CASE WHEN typ = 1 THEN t_create ELSE t_spend END) AS t_rev
                     FROM read_parquet({lj}) WHERE addr IS NOT NULL GROUP BY addr""", rj)
            copy(f"""
                WITH iv AS (
                  SELECT l.addr, {FAM} AS fam, l.sat, l.t_create,
                         CASE WHEN l.t_spend < {t_end} THEN l.t_spend END AS t_spend,
                         l.t_create + INTERVAL ({args.d_min_days}) DAY AS t_dorm,
                         CASE WHEN typ IN (1, 6) THEN l.t_create ELSE r.t_rev END AS t_exp,
                         r.t_rev, {ERA} AS era, {VBAND} AS vband
                  FROM read_parquet({lj}) AS l LEFT JOIN '{rj.as_posix()}' AS r USING (addr)
                  WHERE l.sat >= {args.dust_sat}
                ), ivals AS (
                  SELECT 'exposed' AS pop, addr, fam, era, vband, sat, t_create, t_spend,
                         greatest(t_exp, t_dorm) AS a, coalesce(t_spend, {t_end}) AS b
                  FROM iv WHERE t_exp IS NOT NULL AND t_exp < {t_end}
                  UNION ALL
                  SELECT 'control', addr, fam, era, vband, sat, t_create, t_spend,
                         t_dorm, least(coalesce(t_spend, {t_end}), coalesce(t_rev, {t_end}))
                  FROM iv WHERE fam IN ('p2pkh', 'p2sh')
                )
                SELECT pop, addr, fam, era, vband, sat, t_spend, band,
                       greatest(a, t_create + to_days(d_lo)) AS a,
                       least(b, t_create + to_days(d_hi)) AS b
                FROM ivals CROSS JOIN {bands}
                WHERE a < b
                  AND greatest(a, t_create + to_days(d_lo)) < least(b, t_create + to_days(d_hi))
            """, sj)
        unit(f"BC_{j}", bc)

    segs = f"'{seg_dir.as_posix()}/*.parquet'"
    ev_where = f"""t_spend IS NOT NULL AND t_spend >= a AND t_spend <= b AND t_spend < {t_end}
                   AND date_trunc('week', a) < date_trunc('week', t_spend)
                   AND (pop = 'exposed' OR t_spend = b)"""


    unit("D_risk", lambda: copy(f"""
        WITH deltas AS (
          SELECT pop, fam, era, vband, band, CAST(date_trunc('week', a) AS DATE) + 7 AS wk,
                 1 AS dn, sat AS dv, (sat / 1e8) * (sat / 1e8) AS dv2
          FROM read_parquet({segs})
          UNION ALL
          SELECT pop, fam, era, vband, band, CAST(date_trunc('week', b) AS DATE) + 7,
                 -1, -sat, -(sat / 1e8) * (sat / 1e8)
          FROM read_parquet({segs})
        ), agg AS (
          SELECT pop, fam, era, vband, band, wk, sum(dn) AS dn, sum(dv) AS dv, sum(dv2) AS dv2
          FROM deltas GROUP BY ALL
        )
        SELECT pop, fam, era, vband, band, wk,
               sum(dn) OVER w AS n, sum(dv) OVER w AS sv, sum(dv2) OVER w AS sv2
        FROM agg
        WINDOW w AS (PARTITION BY pop, fam, era, vband, band ORDER BY wk
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    """, res / "risk_weekly.parquet"))


    for s in sorted(seg_dir.glob("*.parquet")):
        if s.name.endswith(".part.parquet"):
            continue
        j = s.stem
        sp = f"'{s.as_posix()}'"

        def efg(sp=sp, j=j):
            copy(f"""SELECT pop, fam, era, vband, band, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
                            count(*) AS n, sum(sat) AS sv, sum((sat / 1e8) * (sat / 1e8)) AS sv2,
                            count(DISTINCT addr) AS n_addr
                     FROM {sp} WHERE {ev_where} GROUP BY ALL""", part_dir / f"E_{j}.parquet")
            copy(f"""WITH w AS (
                       SELECT pop, addr, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
                              sum(sat) AS v, count(*) AS n_out, min(era) AS era, any_value(fam) AS fam
                       FROM {sp} WHERE {ev_where} GROUP BY pop, addr, wk)
                     SELECT pop, fam, era, wk, count(*) AS n_units, sum(v) AS sv,
                            sum((v / 1e8) * (v / 1e8)) AS sv2, max(v) AS vmax
                     FROM w GROUP BY ALL""", part_dir / f"F_{j}.parquet")
            copy(f"""SELECT pop, addr, CAST(date_trunc('week', t_spend) AS DATE) AS wk,
                            sum(sat) AS sat, count(*) AS n_out, min(era) AS era, any_value(fam) AS fam,
                            max(band) AS band
                     FROM {sp} WHERE {ev_where} AND pop = 'exposed'
                     GROUP BY pop, addr, wk
                     HAVING sum(sat) >= {args.big_btc} * 100000000""", part_dir / f"G_{j}.parquet")
        unit(f"EFG_{j}", efg)

    pd_ = part_dir.as_posix()
    unit("E_events", lambda: copy(f"""
        SELECT pop, fam, era, vband, band, wk, sum(n) AS n, sum(sv) AS sv, sum(sv2) AS sv2,
               sum(n_addr) AS n_addr
        FROM read_parquet('{pd_}/E_*.parquet') GROUP BY ALL""", res / "events_weekly.parquet"))
    unit("F_wakeups", lambda: copy(f"""
        SELECT pop, fam, era, wk, sum(n_units) AS n_units, sum(sv) AS sv, sum(sv2) AS sv2,
               max(vmax) AS vmax
        FROM read_parquet('{pd_}/F_*.parquet') GROUP BY ALL""", res / "wakeups_weekly.parquet"))
    unit("G_big", lambda: copy(f"SELECT * FROM read_parquet('{pd_}/G_*.parquet')",
                               res / "big_events.parquet"))
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
