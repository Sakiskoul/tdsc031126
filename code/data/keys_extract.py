from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

BUCKET = "aws-public-blockchain/v1.0/btc/transactions/"
COLS = ["block_timestamp",
        "inputs.list.element.script_hex", "inputs.list.element.spent_transaction_hash",
        "inputs.list.element.txinwitness.list.element",
        "outputs.list.element.type", "outputs.list.element.script_hex"]

KEY_SQL = """
WITH ins AS (
    SELECT CAST(d.block_timestamp AS TIMESTAMP) AS ts, lower(coalesce(i.script_hex, '')) AS s,
           {wit} AS w
    FROM day AS d, UNNEST(d.inputs) AS u(i)
    WHERE i.spent_transaction_hash IS NOT NULL
), outs AS (
    SELECT CAST(d.block_timestamp AS TIMESTAMP) AS ts, lower(o.script_hex) AS s,
           CAST([] AS VARCHAR[]) AS w
    FROM day AS d, UNNEST(d.outputs) AS u(o)
    WHERE o.type IN ('pubkey', 'multisig')
), cand AS (
    SELECT ts, unnest(list_concat(
               regexp_extract_all(s, '21(0[23][0-9a-f]{{64}})', 1),
               regexp_extract_all(s, '41(04[0-9a-f]{{128}})', 1),
               list_filter(w, x -> regexp_full_match(lower(x), '0[23][0-9a-f]{{64}}|04[0-9a-f]{{128}}')),
               regexp_extract_all(lower(array_to_string(w, ' ')), '21(0[23][0-9a-f]{{64}})', 1))) AS k
    FROM (SELECT * FROM ins UNION ALL SELECT * FROM outs)
)
SELECT unhex(CASE WHEN length(k) = 130
                  THEN (CASE WHEN ('0x' || right(k, 1))::INTEGER % 2 = 0 THEN '02' ELSE '03' END)
                       || substr(k, 3, 64)
                  ELSE k END) AS pk,
       max(CASE WHEN length(k) = 130 THEN unhex(k) END) AS pku,
       min(ts) AS ts
FROM cand WHERE k IS NOT NULL
GROUP BY 1
"""


def day_files(s3: pafs.S3FileSystem, date: str) -> list[str]:
    sel = pafs.FileSelector(BUCKET + f"date={date}", allow_not_found=True)
    return sorted(i.path for i in s3.get_file_info(sel)
                  if i.type == pafs.FileType.File and i.path.endswith(".parquet"))


def keys_of(con: duckdb.DuckDBPyConnection, s3, path: str) -> pa.Table:
    pf = pq.ParquetFile(path, filesystem=s3)
    have = {pf.metadata.schema.column(i).path for i in range(pf.metadata.num_columns)}
    tbl = pf.read(columns=[c for c in COLS if c in have])
    i_fields = (set(tbl.schema.field("inputs").type.value_type.names)
                if "inputs" in tbl.schema.names else set())
    wit = ("CAST(coalesce(i.txinwitness, []) AS VARCHAR[])" if "txinwitness" in i_fields
           else "CAST([] AS VARCHAR[])")
    if "inputs" not in tbl.schema.names:
        tbl = tbl.append_column("inputs", pa.array([[]] * tbl.num_rows, pa.list_(
            pa.struct([("script_hex", pa.string()), ("spent_transaction_hash", pa.string())]))))
    con.register("day", tbl)
    out = con.execute(KEY_SQL.format(wit=wit)).fetch_arrow_table()
    con.unregister("day")
    return out


def work(date: str, out_dir: Path) -> str:
    path = out_dir / "keys" / f"date={date}.parquet"
    if path.exists():
        return f"{date} skip"
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-2")
    con = duckdb.connect()
    con.execute("SET threads TO 2")
    for attempt in range(5):
        try:
            t0 = time.time()
            files = day_files(s3, date)
            if not files:
                return f"{date} missing"
            parts = [keys_of(con, s3, f) for f in files]
            con.register("parts", pa.concat_tables(parts))
            tbl = con.execute("SELECT pk, max(pku) AS pku, min(ts) AS ts FROM parts GROUP BY pk").fetch_arrow_table()
            con.unregister("parts")
            tmp = path.with_suffix(".part")
            pq.write_table(tbl, tmp, compression="zstd")
            tmp.rename(path)
            return f"{date} ok keys={tbl.num_rows} files={len(files)} t={time.time() - t0:.1f}s"
        except Exception:
            if attempt == 4:
                return f"{date} FAILED " + traceback.format_exc().replace(chr(10), " | ")
            time.sleep(10 * (attempt + 1))
    return f"{date} FAILED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2009-01-03")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    (out_dir / "keys").mkdir(parents=True, exist_ok=True)
    d0, d1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    dates = [(d0 + dt.timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
            open(out_dir / "keys_extract.log", "a", encoding="utf-8") as fh:
        futs = {ex.submit(work, d, out_dir): d for d in dates}
        for k, fut in enumerate(as_completed(futs), 1):
            fh.write(f"{dt.datetime.utcnow().isoformat()} [{k}/{len(dates)}] {fut.result()}\n")
            fh.flush()
    print(f"done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
