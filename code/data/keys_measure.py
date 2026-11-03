from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

K = 64
T_END = "TIMESTAMP '2026-10-01 00:00:00'"
DUST = 100_000
TRAIN = ("2014-01-06", "2019-12-30")
EVAL = ("2020-01-06", "2026-09-28")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="~/argus_keys")
    ap.add_argument("--life", default="~/hypnos_data/work/life")
    ap.add_argument("--out", default="~/argus_keys/out")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--memory", default="48GB")
    args = ap.parse_args()
    keys = Path(args.keys).expanduser()
    life = Path(args.life).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{(keys / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")


    altb = keys / "alt_by_ap"
    if not (keys / "_done" / "alt_by_ap").exists():
        t0 = time.time()
        con.execute(f"""COPY (SELECT key, min(t_alt) AS t_alt, arg_min(form, t_alt) AS form,
                                     CAST(hash(key) % {K} AS INTEGER) AS ap
                              FROM read_parquet('{(keys / 'alt').as_posix()}/*.parquet') GROUP BY key)
                        TO '{altb.as_posix()}' (FORMAT PARQUET, PARTITION_BY (ap), OVERWRITE_OR_IGNORE true,
                           COMPRESSION ZSTD)""")
        (keys / "_done" / "alt_by_ap").touch()
        print(f"alt_by_ap {time.time() - t0:.0f}s", flush=True)


    rows = out / "control_rows"
    rows.mkdir(exist_ok=True)
    for j in range(K):
        dst = rows / f"{j}.parquet"
        if dst.exists():
            continue
        t0 = time.time()
        alt = altb / f"ap={j}"
        alt_src = f"read_parquet('{alt.as_posix()}/*.parquet')" if alt.exists() else \
            "(SELECT NULL::BLOB AS key, NULL::TIMESTAMP AS t_alt, NULL::TINYINT AS form WHERE false)"
        con.execute(f"""COPY (
            WITH L AS (
                SELECT key, typ, sat, t_create, CASE WHEN t_spend < {T_END} THEN t_spend END AS t_spend
                FROM read_parquet('{(life / f'ap={j}').as_posix()}/*.parquet') WHERE t_create < {T_END}),
            R AS (SELECT key, min(CASE WHEN typ = 1 THEN t_create ELSE t_spend END) AS t_rev
                  FROM L GROUP BY key),
            A AS (SELECT key, t_alt, form FROM {alt_src}),
            C AS (
                SELECT L.typ, L.sat, L.t_create, L.t_spend, R.t_rev, A.t_alt, A.form,
                       L.t_create + INTERVAL 365 DAY AS a,
                       least(coalesce(L.t_spend, {T_END}), coalesce(R.t_rev, {T_END})) AS b
                FROM L JOIN R USING (key) LEFT JOIN A USING (key)
                WHERE L.typ IN (2, 3, 4, 5) AND L.sat >= {DUST})
            SELECT typ, sat, a, b, t_alt, form,
                   (t_spend IS NOT NULL AND t_spend = b AND date_trunc('week', a) < date_trunc('week', t_spend))
                       AS event,
                   t_alt IS NOT NULL AND t_alt < b AS hit
            FROM C WHERE a < b)
            TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        print(f"rows {j} {time.time() - t0:.0f}s", flush=True)

    R = f"read_parquet('{rows.as_posix()}/*.parquet')"
    res = {}

    def share(where: str = "true") -> dict:
        q = con.execute(f"""
            SELECT sum(epoch(b) - epoch(a)) / 86400 AS pd,
                   sum((epoch(b) - epoch(a)) * sat / 1e8) / 86400 AS vd,
                   sum(CASE WHEN hit THEN epoch(b) - epoch(greatest(a, t_alt)) ELSE 0 END) / 86400 AS pd_hit,
                   sum(CASE WHEN hit THEN (epoch(b) - epoch(greatest(a, t_alt))) * sat / 1e8 ELSE 0 END) / 86400 AS vd_hit,
                   count(*) FILTER (WHERE event) AS ev, sum(sat) FILTER (WHERE event) / 1e8 AS ev_btc,
                   count(*) FILTER (WHERE event AND hit) AS ev_hit,
                   sum(sat) FILTER (WHERE event AND hit) / 1e8 AS ev_hit_btc,
                   count(*) AS outputs, count(*) FILTER (WHERE hit) AS outputs_hit
            FROM {R} WHERE {where}""").fetchone()
        names = ["person_days", "btc_days", "person_days_hit", "btc_days_hit", "events", "events_btc",
                 "events_hit", "events_hit_btc", "outputs", "outputs_hit"]
        d = dict(zip(names, [float(x or 0) for x in q]))
        for a, b in (("person_days_hit", "person_days"), ("btc_days_hit", "btc_days"),
                     ("events_hit", "events"), ("events_hit_btc", "events_btc"), ("outputs_hit", "outputs")):
            d[a.replace("_hit", "") + "_share"] = d[a] / d[b] if d[b] else None
        return d

    res["all"] = share()
    for name, (s, e) in (("train", TRAIN), ("eval", EVAL)):

        ev = con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE hit), coalesce(sum(sat), 0) / 1e8,
                   coalesce(sum(sat) FILTER (WHERE hit), 0) / 1e8
            FROM {R} WHERE event AND b >= TIMESTAMP '{s}' AND b < TIMESTAMP '{e}' + INTERVAL 7 DAY""").fetchone()
        pt = con.execute(f"""
            WITH w AS (SELECT sat, hit, t_alt, greatest(a, TIMESTAMP '{s}') AS a2,
                              least(b, TIMESTAMP '{e}' + INTERVAL 7 DAY) AS b2 FROM {R})
            SELECT coalesce(sum(epoch(b2) - epoch(a2)), 0),
                   coalesce(sum(CASE WHEN hit AND t_alt < b2 THEN epoch(b2) - epoch(greatest(a2, t_alt)) ELSE 0 END), 0),
                   coalesce(sum((epoch(b2) - epoch(a2)) * sat), 0),
                   coalesce(sum(CASE WHEN hit AND t_alt < b2
                                     THEN (epoch(b2) - epoch(greatest(a2, t_alt))) * sat ELSE 0 END), 0)
            FROM w WHERE a2 < b2""").fetchone()
        res[name] = {"events": ev[0], "events_hit": ev[1], "events_share": ev[1] / ev[0] if ev[0] else None,
                     "events_btc": ev[2], "events_hit_btc": ev[3],
                     "events_btc_share": ev[3] / ev[2] if ev[2] else None,
                     "person_time_share": pt[1] / pt[0] if pt[0] else None,
                     "btc_time_share": pt[3] / pt[2] if pt[2] else None}
    res["by_form"] = {int(f): {"outputs": n, "events": e, "events_btc": v} for f, n, e, v in con.execute(f"""
        SELECT form, count(*), count(*) FILTER (WHERE event), sum(sat) FILTER (WHERE event) / 1e8
        FROM {R} WHERE hit GROUP BY form ORDER BY form""").fetchall()}
    res["by_type"] = {int(t): {"outputs": n, "outputs_hit": h, "events": e, "events_hit": eh}
                      for t, n, h, e, eh in con.execute(f"""
        SELECT typ, count(*), count(*) FILTER (WHERE hit), count(*) FILTER (WHERE event),
               count(*) FILTER (WHERE event AND hit) FROM {R} GROUP BY typ ORDER BY typ""").fetchall()}
    res["events_by_year"] = {int(y): {"events": e, "events_hit": h, "btc": v, "btc_hit": vh}
                             for y, e, h, v, vh in con.execute(f"""
        SELECT year(b), count(*), count(*) FILTER (WHERE hit), sum(sat) / 1e8,
               coalesce(sum(sat) FILTER (WHERE hit), 0) / 1e8
        FROM {R} WHERE event GROUP BY 1 ORDER BY 1""").fetchall()}
    con.execute(f"""COPY (SELECT CAST(date_trunc('week', b) AS DATE) AS wk, count(*) AS n, count(*) FILTER (WHERE hit) AS n_hit,
                                 sum(sat) / 1e8 AS btc, coalesce(sum(sat) FILTER (WHERE hit), 0) / 1e8 AS btc_hit
                          FROM {R} WHERE event GROUP BY 1 ORDER BY 1)
                    TO '{(out / 'weekly.parquet').as_posix()}' (FORMAT PARQUET)""")
    (out / "contamination.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["all"], indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("train", "eval")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
