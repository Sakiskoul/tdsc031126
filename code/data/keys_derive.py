from __future__ import annotations

import argparse
import hashlib
import sys
import time
from multiprocessing import Pool
from pathlib import Path

P = 2 ** 256 - 2 ** 32 - 977
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
GEN = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
PARTS = 32


def h160(b: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(b).digest()).digest()


def b58check(payload: bytes) -> str:
    data = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n, out = int.from_bytes(data, "big"), []
    while n:
        n, r = divmod(n, 58)
        out.append(B58[r])
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + "".join(reversed(out))


def _polymod(values) -> int:
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
    return chk


def p2wpkh(h: bytes) -> str:
    acc, bits, data = 0, 0, [0]
    for x in h:
        acc = (acc << 8) | x
        bits += 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits:
        data.append((acc << (5 - bits)) & 31)
    chk = _polymod([3, 3, 0, 2, 3] + data + [0] * 6) ^ 1
    return "bc1" + "".join(BECH[d] for d in data + [(chk >> 5 * (5 - i)) & 31 for i in range(6)])


def uncompressed(pk: bytes) -> bytes | None:
    x = int.from_bytes(pk[1:], "big")
    if x >= P:
        return None
    rhs = (pow(x, 3, P) + 7) % P
    y = pow(rhs, (P + 1) // 4, P)
    if y * y % P != rhs:
        return None
    if y % 2 != pk[0] - 2:
        y = P - y
    return b"\x04" + pk[1:] + y.to_bytes(32, "big")


def addresses(pk: bytes, pku: bytes | None) -> list[tuple[str, int]]:
    if pku is None:
        pku = uncompressed(pk)
        if pku is None:
            return []
    h = h160(pk)
    return [(b58check(b"\x00" + h), 1), (b58check(b"\x00" + h160(pku)), 2),
            (p2wpkh(h), 3), (b58check(b"\x05" + h160(b"\x00\x14" + h)), 4)]


KEY = "unhex(md5(a))"


def derive_part(args) -> str:

    j, keys_dir, done, batch, cap_gb, duck_mem, max_batches = args
    import resource
    import shutil
    cap = int(cap_gb * 2 ** 30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    import duckdb
    import pyarrow as pa
    import pyarrow.dataset as pds
    import pyarrow.parquet as pq
    mark = Path(done) / f"derive_{j}"
    if mark.exists():
        return f"derive {j} skip"
    t0 = time.time()
    tmpdir = Path(keys_dir) / "alt_tmp" / str(j)
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True)
    n_keys = n_addr = 0
    src = pds.dataset(Path(keys_dir) / "uniq" / f"p={j}", format="parquet")
    for b, rb in enumerate(src.to_batches(columns=["pk", "pku", "ts"], batch_size=batch)):
        if max_batches and b >= max_batches:
            break
        a, form, ts = [], [], []
        for pk, pku, t in zip(rb.column(0).to_pylist(), rb.column(1).to_pylist(), rb.column(2).to_pylist()):
            for s, f in addresses(pk, pku):
                a.append(s)
                form.append(f)
                ts.append(t)
        pq.write_table(pa.table({"a": pa.array(a, pa.string()), "form": pa.array(form, pa.int8()),
                                 "ts": pa.array(ts, pa.timestamp("us"))}),
                       tmpdir / f"{b}.parquet", compression="zstd")
        n_keys += rb.num_rows
        n_addr += len(a)
        del a, form, ts, rb
    con = duckdb.connect()
    con.execute("SET threads TO 1")
    con.execute(f"SET memory_limit = '{duck_mem}'")
    con.execute(f"SET temp_directory = '{(tmpdir / 'spill').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    out = Path(keys_dir) / "alt" / f"{j}.parquet"
    tmp = out.with_suffix(".part")
    con.execute(f"""COPY (SELECT {KEY} AS key, min(ts) AS t_alt, arg_min(form, ts) AS form
                          FROM read_parquet('{tmpdir.as_posix()}/*.parquet') GROUP BY 1)
                    TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.close()
    if max_batches:
        tmp.unlink()
        shutil.rmtree(tmpdir, ignore_errors=True)
        return f"dry {j} keys={n_keys} addrs={n_addr} t={time.time() - t0:.0f}s"
    tmp.rename(out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    mark.touch()
    return f"derive {j} keys={n_keys} addrs={n_addr} t={time.time() - t0:.0f}s"


def merge(keys_dir: Path, done: Path, threads: int, memory: str) -> None:
    import duckdb
    mark = done / "merge"
    if mark.exists():
        return
    t0 = time.time()
    con = duckdb.connect()
    con.execute(f"SET threads TO {threads}")
    con.execute(f"SET memory_limit = '{memory}'")
    con.execute(f"SET temp_directory = '{(keys_dir / 'tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""COPY (SELECT pk, max(pku) AS pku, min(ts) AS ts, CAST(hash(pk) % {PARTS} AS INTEGER) AS p
                          FROM read_parquet('{(keys_dir / 'keys').as_posix()}/*.parquet') GROUP BY pk)
                    TO '{(keys_dir / 'uniq').as_posix()}' (FORMAT PARQUET, PARTITION_BY (p),
                       OVERWRITE_OR_IGNORE true, COMPRESSION ZSTD)""")
    mark.touch()
    print(f"merge {time.time() - t0:.0f}s", flush=True)


def selftest(date: str) -> int:

    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-2")
    sel = pafs.FileSelector("aws-public-blockchain/v1.0/btc/transactions/" + f"date={date}")
    path = sorted(i.path for i in s3.get_file_info(sel) if i.path.endswith(".parquet"))[0]
    pf = pq.ParquetFile(path, filesystem=s3)
    have = {pf.metadata.schema.column(i).path for i in range(pf.metadata.num_columns)}
    cols = [c for c in ("inputs.list.element.type", "inputs.list.element.address",
                        "inputs.list.element.script_hex", "inputs.list.element.txinwitness.list.element")
            if c in have]
    rows = pf.read_row_group(0, columns=cols).column("inputs").to_pylist()
    stats = {}
    for tx in rows:
        for i in tx or []:
            typ, addr, s = i["type"], i["address"], (i["script_hex"] or "").lower()
            w = i.get("txinwitness") or []
            if typ == "pubkeyhash" and len(s) > 132 and s[-132:-128] == "4104":
                k = bytes.fromhex(s[-130:])
            elif typ == "pubkeyhash" and len(s) > 68 and s[-68:-66] == "21" and s[-66:-64] in ("02", "03"):
                k = bytes.fromhex(s[-66:])
            elif typ in ("witness_v0_keyhash", "scripthash") and len(w) == 2 and len(w[1]) == 66:
                k = bytes.fromhex(w[1])
            else:
                continue
            pk = k if len(k) == 33 else bytes([2 + k[-1] % 2]) + k[1:33]
            pku = k if len(k) == 65 else None
            derived = {f: a for a, f in addresses(pk, pku)}
            form = {"pubkeyhash": 2 if pku else 1, "witness_v0_keyhash": 3, "scripthash": 4}[typ]
            if typ == "scripthash" and not s.startswith("160014"):
                continue
            ok = derived.get(form) == addr
            stats.setdefault(form, [0, 0])[0 if ok else 1] += 1
    print(date, {f: {"match": m, "miss": x} for f, (m, x) in sorted(stats.items())})
    return 0 if stats and all(x == 0 for m, x in stats.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="~/argus_keys")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="16GB", help="DuckDB limit of the merge")
    ap.add_argument("--batch", type=int, default=500_000, help="keys per batch in a worker")
    ap.add_argument("--cap-gb", type=float, default=8.0, help="address-space cap of each worker")
    ap.add_argument("--duck-mem", default="2GB", help="DuckDB limit inside each worker")
    ap.add_argument("--dry-run", type=int, default=0, help="process only this many batches of one part")
    ap.add_argument("--part", type=int, default=None, help="the part of a dry run")
    ap.add_argument("--selftest", default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.selftest)
    keys_dir = Path(args.keys).expanduser()
    done = keys_dir / "_done"
    for d in (done, keys_dir / "alt", keys_dir / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    merge(keys_dir, done, args.threads, args.memory)
    common = (str(keys_dir), str(done), args.batch, args.cap_gb, args.duck_mem, args.dry_run)
    if args.dry_run:
        print(derive_part((args.part or 0, *common)), flush=True)
        return 0
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(derive_part, [(j, *common) for j in range(PARTS)]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
