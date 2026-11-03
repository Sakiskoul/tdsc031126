from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

BUCKET = "aws-public-blockchain/v1.0/eth/logs/"
POOLS = {
    0.1: "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc",
    1.0: "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936",
    10.0: "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf",
    100.0: "0xa160cdab225685da1d56aa342ad8841c3b53f291",
}
DATA_LEN = {2 + 128: "deposit", 2 + 192: "withdrawal"}


def work(date: str, out_dir: Path) -> str:
    out = out_dir / f"date={date}.parquet"
    if out.exists():
        return f"{date} skip"
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-2")
    for attempt in range(5):
        try:
            sel = pafs.FileSelector(BUCKET + f"date={date}", allow_not_found=True)
            files = sorted(i.path for i in s3.get_file_info(sel) if i.path.endswith(".parquet"))
            rows = []
            for f in files:
                t = pq.ParquetFile(f, filesystem=s3).read(columns=["address", "data", "block_timestamp"])
                addr = pc.utf8_lower(t["address"])
                mask = pc.is_in(addr, value_set=pa.array(list(POOLS.values())))
                sub = t.filter(mask)
                for a, d, ts in zip(pc.utf8_lower(sub["address"]).to_pylist(), sub["data"].to_pylist(),
                                    sub["block_timestamp"].to_pylist()):
                    kind = DATA_LEN.get(len(d or ""), "other")
                    denom = next(k for k, v in POOLS.items() if v == a)
                    rows.append((ts, denom, kind))
            tbl = pa.table({"ts": pa.array([r[0] for r in rows], pa.timestamp("us")),
                            "denom": pa.array([r[1] for r in rows], pa.float64()),
                            "kind": pa.array([r[2] for r in rows], pa.string())})
            pq.write_table(tbl, out)
            return f"{date} ok files={len(files)} events={len(rows)}"
        except Exception:
            out.unlink(missing_ok=True)
            if attempt == 4:
                return f"{date} FAILED " + traceback.format_exc().replace(chr(10), " | ")
            time.sleep(10 * (attempt + 1))
    return f"{date} FAILED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2019-12-15")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    d0, d1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    dates = [(d0 + dt.timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(out_dir / "extract.log", "a") as fh:
        futs = [ex.submit(work, d, out_dir) for d in dates]
        for k, fut in enumerate(as_completed(futs), 1):
            fh.write(f"[{k}/{len(dates)}] {fut.result()}\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
