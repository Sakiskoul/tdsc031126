import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code" / "data" / "person_time.py"


def h16(s: str) -> bytes:
    return hashlib.md5(s.encode()).digest()


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def build(tmp: Path) -> None:
    (tmp / "outputs").mkdir(parents=True)
    (tmp / "inputs").mkdir(parents=True)
    outs = [
        (h16("t1"), 0, ts("2015-01-01"), 2, h16("A"), 100_000_000),
        (h16("t1"), 1, ts("2015-01-01"), 2, h16("A"), 200_000_000),
        (h16("t2"), 0, ts("2010-01-01"), 1, h16("B"), 5_000_000_000),
        (h16("t3"), 0, ts("2014-01-01"), 2, h16("C"), 300_000_000),
    ]
    ins = [
        (h16("t1"), 0, ts("2016-06-01")),
        (h16("t1"), 1, ts("2018-03-01")),
        (h16("t2"), 0, ts("2025-07-01")),
    ]
    o = pa.table({
        "tx": pa.array([r[0] for r in outs], pa.binary(16)),
        "vout": pa.array([r[1] for r in outs], pa.uint32()),
        "ts": pa.array([r[2] for r in outs], pa.timestamp("us")),
        "typ": pa.array([r[3] for r in outs], pa.uint8()),
        "addr": pa.array([r[4] for r in outs], pa.binary(16)),
        "sat": pa.array([r[5] for r in outs], pa.int64()),
    })
    i = pa.table({
        "ptx": pa.array([r[0] for r in ins], pa.binary(16)),
        "pvout": pa.array([r[1] for r in ins], pa.uint32()),
        "ts": pa.array([r[2] for r in ins], pa.timestamp("us")),
    })
    pq.write_table(o, tmp / "outputs" / "date=x.parquet")
    pq.write_table(i, tmp / "inputs" / "date=x.parquet")


def test_exposure_semantics(tmp_path):
    build(tmp_path)
    r = subprocess.run([sys.executable, str(SCRIPT), "--data", str(tmp_path), "--threads", "2",
                        "--memory", "1GB", "--big-btc", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = duckdb.connect()
    res = tmp_path / "results"
    ev = c.execute(f"SELECT pop, fam, band, wk, n, sv FROM '{res}/events_weekly.parquet' ORDER BY wk").fetchall()
    assert ev == [
        ("control", "p2pkh", "a1_2", dt.date(2016, 5, 30), 1, 100_000_000),
        ("exposed", "p2pkh", "a2_5", dt.date(2018, 2, 26), 1, 200_000_000),
        ("exposed", "p2pk", "a10_", dt.date(2025, 6, 30), 1, 5_000_000_000),
    ]

    def at_risk(week: str, pop: str) -> int:
        q = f"""SELECT coalesce(sum(n), 0) FROM (
                  SELECT pop, fam, era, vband, band, last(n ORDER BY wk) AS n
                  FROM '{res}/risk_weekly.parquet'
                  WHERE wk <= DATE '{week}' AND pop = '{pop}' GROUP BY ALL)"""
        return int(c.execute(q).fetchone()[0])


    assert at_risk("2017-01-02", "exposed") == 2
    assert at_risk("2017-01-02", "control") == 1

    assert at_risk("2015-06-01", "exposed") == 1
    assert at_risk("2015-06-01", "control") == 1

    assert at_risk("2019-01-07", "exposed") == 1
    assert at_risk("2019-01-07", "control") == 1
    big = c.execute(f"SELECT count(*) FROM '{res}/big_events.parquet'").fetchone()[0]
    assert big == 2
