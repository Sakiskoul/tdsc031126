from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import duckdb

from person_time_addr import VB, ev_where

LAWS = "(VALUES (0, 'tilted'), (1, 'mimic'), (2, 'uniform'), (3, 'whale')) AS laws(lid, law)"


def export_bucket(con, seg: str, out: Path, stem: str, kb: int, t_end: str, draws: int) -> None:
    con.execute(f"CREATE OR REPLACE TEMP VIEW s AS SELECT * FROM read_parquet('{seg}') WHERE pop = 'exposed'")
    con.execute("""CREATE OR REPLACE TEMP TABLE meta AS
                   SELECT addr, CAST(hash(addr) >> 1 AS BIGINT) AS id, min(era) AS era, min(fam) AS fam
                   FROM s GROUP BY addr""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE x AS
        WITH d AS (
          SELECT addr, a AS ts, sat AS dv FROM s
          UNION ALL
          SELECT addr, b AS ts, -sat AS dv FROM s
        ), c AS (
          SELECT addr, ts, CAST(sum(dv) AS BIGINT) AS dv FROM d GROUP BY addr, ts
        )
        SELECT addr, ts, CAST(sum(dv) OVER (PARTITION BY addr ORDER BY ts
                                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS BIGINT) AS v_new
        FROM c WHERE dv <> 0""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE v0 AS
        SELECT st.sid, x.addr, arg_max(x.v_new, x.ts) AS v0
        FROM x JOIN st ON x.ts < st.t0 GROUP BY st.sid, x.addr""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pool AS
        WITH c AS (
          SELECT v.sid, m.addr, m.id, v.v0, m.fam, m.era, {VB.format(v='v.v0')} AS vband
          FROM v0 v JOIN meta m USING (addr) WHERE v.v0 > 0
        ), w AS (
          SELECT c.*, laws.lid, laws.law, dr.draw,
                 CASE laws.law WHEN 'tilted' THEN coalesce(l.lam, 0) * c.v0 / 1e8
                               WHEN 'mimic' THEN coalesce(l.lam, 0)
                               WHEN 'uniform' THEN 1.0
                               ELSE CASE WHEN c.v0 >= 10000000000 THEN 1.0 ELSE 0.0 END END AS wt,
                 (CAST(hash(c.addr, c.sid, laws.lid, dr.draw) >> 11 AS DOUBLE) + 0.5) / 9007199254740992.0 AS u
          FROM c CROSS JOIN {LAWS} CROSS JOIN (SELECT range AS draw FROM range({draws})) AS dr
          LEFT JOIN lam l ON l.sid = c.sid AND l.fam = c.fam AND l.era = c.era AND l.vband = c.vband
        ), top AS (
          SELECT sid, law, draw,
                 min_by({{'addr': addr, 'id': id, 'key': -ln(u) / wt, 'v0': v0, 'fam': fam, 'era': era}},
                        -ln(u) / wt, {kb}) AS t
          FROM w WHERE wt > 0 GROUP BY sid, law, draw
        )
        SELECT sid, law, CAST(draw AS INTEGER) AS draw, unnest(t, recursive := true) FROM top""")
    con.execute(f"COPY (SELECT sid, law, draw, id, key, v0, fam, era FROM pool) TO '{(out / f'pool_{stem}.parquet').as_posix()}' (FORMAT PARQUET)")
    con.execute("CREATE OR REPLACE TEMP TABLE pa AS SELECT DISTINCT sid, addr, id FROM pool")
    con.execute(f"""
        COPY (
          SELECT pa.sid, pa.id, x.ts, x.v_new
          FROM pa JOIN st ON pa.sid = st.sid JOIN x ON x.addr = pa.addr AND x.ts >= st.t0 AND x.ts < st.t1
        ) TO '{(out / f'chg_{stem}.parquet').as_posix()}' (FORMAT PARQUET)""")
    con.execute(f"""
        COPY (
          SELECT DISTINCT pa.sid, pa.id, CAST(date_trunc('week', s.t_spend) AS DATE) AS wk
          FROM s JOIN pa ON s.addr = pa.addr JOIN st ON pa.sid = st.sid
          WHERE {ev_where(t_end)} AND s.t_spend >= st.t0 AND s.t_spend < st.t1
        ) TO '{(out / f'ev_{stem}.parquet').as_posix()}' (FORMAT PARQUET)""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--starts", required=True, help="csv with sid, t0 (week start)")
    ap.add_argument("--lam", required=True, help="csv with sid, fam, era, vband, lam")
    ap.add_argument("--out", default="replay")
    ap.add_argument("--k", type=int, default=64000, help="global pool size per start, law and draw")
    ap.add_argument("--draws", type=int, default=8, help="independent target draws per start and law")
    ap.add_argument("--t-end", default="2026-10-01")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--memory", default="10GB")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    out = data / args.out
    done = out / "_done"
    for p in (out, done, data / "tmp"):
        p.mkdir(parents=True, exist_ok=True)
    kb = math.ceil(1.5 * args.k / 64) + 100
    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{(data / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""CREATE TABLE st AS SELECT sid, CAST(t0 AS TIMESTAMP) AS t0,
                    CAST(t0 AS TIMESTAMP) + INTERVAL 364 DAY AS t1
                    FROM read_csv('{Path(args.starts).expanduser().as_posix()}', header = true)""")
    con.execute(f"""CREATE TABLE lam AS SELECT * FROM read_csv('{Path(args.lam).expanduser().as_posix()}', header = true)""")
    t_end = f"TIMESTAMP '{args.t_end} 00:00:00'"
    for seg in sorted((data / "work" / "seg").glob("*.parquet")):
        if seg.name.endswith(".part.parquet"):
            continue
        mark = done / f"REPLAY_{seg.stem}"
        if mark.exists():
            continue
        t = time.time()
        export_bucket(con, seg.as_posix(), out, seg.stem, kb, t_end, args.draws)
        mark.touch()
        print(f"[REPLAY_{seg.stem}] {time.time() - t:.0f}s", flush=True)
    (out / "meta.txt").write_text(f"k={args.k} kb={kb} draws={args.draws}\n")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
