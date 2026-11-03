from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa

PUZZLE_TX = "08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15"


def digest_as_stored(con: duckdb.DuckDBPyConnection, value: str) -> bytes:

    con.register("one", pa.table({"s": [value]}))
    t = con.execute("SELECT md5_number(s) AS h FROM one").fetch_arrow_table()
    con.unregister("one")
    return t.column("h")[0].as_py()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="8GB")
    args = ap.parse_args()
    data = Path(args.data).expanduser()
    con = duckdb.connect()

    con.execute(f"SET threads TO {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory}'")
    con.execute(f"SET temp_directory = '{(data / 'tmp').as_posix()}'")
    tx = digest_as_stored(con, PUZZLE_TX)
    day = data / "outputs" / "date=2015-01-15.parquet"
    puz = con.execute(f"""
        SELECT vout + 1 AS puzzle, addr, sat FROM '{day.as_posix()}'
        WHERE tx = ? AND vout < 160 ORDER BY vout
    """, [tx]).fetchdf()
    print("puzzle outputs found:", len(puz))
    con.register("puz", puz)
    seg = data / args.work / "seg.parquet"
    if not seg.exists():
        seg = data / args.work / "seg" / "*.parquet"
    ev = con.execute(f"""
        SELECT p.puzzle, s.pop, s.band, s.fam, CAST(date_trunc('week', s.t_spend) AS DATE) AS wk,
               s.t_spend, s.sat / 1e8 AS btc, s.a AS risk_start
        FROM '{seg.as_posix()}' s JOIN puz p ON s.addr = p.addr
        WHERE s.t_spend IS NOT NULL AND s.t_spend >= s.a AND s.t_spend <= s.b
        ORDER BY s.t_spend
    """).fetchdf()
    out = data / args.results / "rq4_puzzle_events.csv"
    ev.to_csv(out, index=False)
    print(ev[ev["puzzle"] % 5 == 0].tail(30).to_string())
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
