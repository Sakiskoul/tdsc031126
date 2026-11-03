from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus import attest as A

CACHE = Path(__file__).resolve().parents[2] / "contracts" / ".cache"


def operators(n: int, start: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for op in range(n):
        f = CACHE / f"op{op}_{start}.pkl"
        o = A.Operator(op=op, seed=f"argus-test-operator-{op}".encode(), start_week=start)
        if f.exists():
            levels = pickle.loads(f.read_bytes())
        else:
            levels = o.tree()
            f.write_bytes(pickle.dumps(levels))
        out.append((o, levels))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["roots", "sign"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start-week", type=int, default=2800)
    ap.add_argument("--chain-id", type=int, default=1337)
    ap.add_argument("--contract", default="0x" + "00" * 20)
    ap.add_argument("--week", type=int, default=2805)
    ap.add_argument("--log", default=None, help="write-ahead sign-once log; operators always set it")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--digest", default="0x" + "11" * 32)
    args = ap.parse_args()
    ops = operators(args.n, args.start_week)
    if args.cmd == "roots":
        print(json.dumps({"roots": ["0x" + lv[-1][0].hex() for _, lv in ops]}))
        return 0
    contract = bytes.fromhex(args.contract[2:])
    digest = bytes.fromhex(args.digest[2:])
    m = A.statement(args.chain_id, contract, args.week, args.level, digest)
    idx = args.week - args.start_week
    log = A.SignLog(args.log) if args.log else None
    sigs = []
    for o, lv in ops:
        sig = o.sign_once(m, args.week, log) if log else o.sign(m, args.week)
        sigs.append({"op": o.op, "sig": ["0x" + s.hex() for s in sig],
                     "path": ["0x" + p.hex() for p in A.auth_path(lv, idx)]})
    print(json.dumps({"statement": "0x" + m.hex(), "sigs": sigs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
