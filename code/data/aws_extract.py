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
COLS = ["txid", "block_timestamp",
        "outputs.list.element.address", "outputs.list.element.index",
        "outputs.list.element.type", "outputs.list.element.value",
        "inputs.list.element.spent_transaction_hash", "inputs.list.element.spent_output_index"]

TYPE_CODE = """
CASE o.type
  WHEN 'pubkey' THEN 1 WHEN 'pubkeyhash' THEN 2 WHEN 'witness_v0_keyhash' THEN 3
  WHEN 'scripthash' THEN 4 WHEN 'witness_v0_scripthash' THEN 5 WHEN 'witness_v1_taproot' THEN 6
  ELSE 0 END
"""


SCRIPT_HEX = "outputs.list.element.script_hex"
P2PK_SCRIPT_UNTIL = "2014-01-01"


def day_files(s3: pafs.S3FileSystem, date: str) -> list[str]:
    sel = pafs.FileSelector(BUCKET + f"date={date}", allow_not_found=True)
    return sorted(i.path for i in s3.get_file_info(sel)
                  if i.type == pafs.FileType.File and i.path.endswith(".parquet"))


def read_file(s3: pafs.S3FileSystem, path: str, date: str) -> pa.Table:
    pf = pq.ParquetFile(path, filesystem=s3)
    have = {pf.metadata.schema.column(i).path for i in range(pf.metadata.num_columns)}
    cols = [c for c in COLS if c in have]
    if date < P2PK_SCRIPT_UNTIL and SCRIPT_HEX in have:
        cols.append(SCRIPT_HEX)
    return pf.read(columns=cols)


def flatten(con: duckdb.DuckDBPyConnection, tbl: pa.Table) -> tuple[pa.Table, pa.Table]:

    names = set(tbl.schema.names)
    con.register("day", tbl)
    o_fields = set(tbl.schema.field("outputs").type.value_type.names) if "outputs" in names else set()
    addr = "md5_number(o.address)" if "address" in o_fields else "CAST(NULL AS UHUGEINT)"
    if "script_hex" in o_fields:
        addr = f"CASE WHEN o.type = 'pubkey' THEN md5_number(o.script_hex) ELSE {addr} END"
    outs = con.execute(f"""
        SELECT md5_number(d.txid) AS tx,
               CAST(o.index AS UINTEGER) AS vout,
               CAST(d.block_timestamp AS TIMESTAMP) AS ts,
               CAST({TYPE_CODE} AS UTINYINT) AS typ,
               {addr} AS addr,
               CAST(round(o.value * 1e8) AS BIGINT) AS sat
        FROM day AS d, UNNEST(d.outputs) AS u(o)
        WHERE ({TYPE_CODE}) > 0
    """).fetch_arrow_table()
    if "inputs" in names:
        ins = con.execute("""
            SELECT md5_number(i.spent_transaction_hash) AS ptx,
                   CAST(i.spent_output_index AS UINTEGER) AS pvout,
                   CAST(d.block_timestamp AS TIMESTAMP) AS ts
            FROM day AS d, UNNEST(d.inputs) AS u(i)
            WHERE i.spent_transaction_hash IS NOT NULL
        """).fetch_arrow_table()
    else:
        ins = con.execute("""SELECT CAST(NULL AS UHUGEINT) AS ptx, CAST(NULL AS UINTEGER) AS pvout,
                                    CAST(NULL AS TIMESTAMP) AS ts WHERE false""").fetch_arrow_table()
    con.unregister("day")
    return outs, ins


def work(date: str, out_dir: Path) -> str:
    o_path = out_dir / "outputs" / f"date={date}.parquet"
    i_path = out_dir / "inputs" / f"date={date}.parquet"
    if o_path.exists() and i_path.exists():
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
            outs, ins, n_tx = [], [], 0
            for path in files:
                tbl = read_file(s3, path, date)
                n_tx += tbl.num_rows
                o, i = flatten(con, tbl)
                outs.append(o)
                ins.append(i)
            t1 = time.time()
            o_tbl = pa.concat_tables(outs)
            i_tbl = pa.concat_tables(ins)
            pq.write_table(o_tbl, o_path, compression="zstd")
            pq.write_table(i_tbl, i_path, compression="zstd")
            return (f"{date} ok tx={n_tx} out={o_tbl.num_rows} in={i_tbl.num_rows} "
                    f"read={t1-t0:.1f}s write={time.time()-t1:.1f}s")
        except Exception:
            for p in (o_path, i_path):
                p.unlink(missing_ok=True)
            if attempt == 4:
                return f"{date} FAILED " + traceback.format_exc().replace(chr(10), " | ")
            time.sleep(10 * (attempt + 1))
    return f"{date} FAILED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2009-01-03")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    d0 = dt.date.fromisoformat(args.start)
    d1 = dt.date.fromisoformat(args.end)
    dates = [(d0 + dt.timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]
    log = out_dir / "extract.log"
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(log, "a", encoding="utf-8") as fh:
        futs = {ex.submit(work, d, out_dir): d for d in dates}
        for k, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            fh.write(f"{dt.datetime.utcnow().isoformat()} [{k}/{len(dates)}] {msg}\n")
            fh.flush()
    print(f"done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
