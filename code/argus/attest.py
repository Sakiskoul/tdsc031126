from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
import time

from Crypto.Hash import keccak

LEN1, LEN2, W1, H = 64, 3, 15, 10
LEN = LEN1 + LEN2


def k256(*parts: bytes) -> bytes:
    h = keccak.new(digest_bits=256)
    for p in parts:
        h.update(p)
    return h.digest()


def f(x: bytes, op: int, week: int, i: int, j: int) -> bytes:
    return k256(x, op.to_bytes(8, "big"), week.to_bytes(8, "big"), i.to_bytes(2, "big"), bytes([j]))


def digits(m: bytes) -> list[int]:
    d = []
    for i in range(LEN1):
        b = m[i // 2]
        d.append(b >> 4 if i % 2 == 0 else b & 0x0F)
    csum = sum(W1 - v for v in d)
    d += [(csum >> 8) & 0x0F, (csum >> 4) & 0x0F, csum & 0x0F]
    return d


def statement(chain_id: int, contract: bytes, week: int, level: int, digest: bytes) -> bytes:
    return k256(b"ARGUS-ALARM-v1", chain_id.to_bytes(32, "big"), contract,
                week.to_bytes(32, "big"), bytes([level]), digest)


@dataclass
class Operator:
    op: int
    seed: bytes
    start_week: int

    def _sk(self, week: int, i: int) -> bytes:
        return k256(b"ARGUS-SK", self.seed, week.to_bytes(8, "big"), i.to_bytes(2, "big"))

    def pk(self, week: int) -> bytes:
        ends = []
        for i in range(LEN):
            x = self._sk(week, i)
            for j in range(W1):
                x = f(x, self.op, week, i, j)
            ends.append(x)
        return k256(*ends)

    def leaves(self) -> list[bytes]:
        return [k256(self.pk(self.start_week + t)) for t in range(1 << H)]

    def tree(self) -> list[list[bytes]]:
        levels = [self.leaves()]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            levels.append([k256(cur[2 * t], cur[2 * t + 1]) for t in range(len(cur) // 2)])
        return levels

    def sign(self, m: bytes, week: int) -> list[bytes]:


        sig = []
        for i, b in enumerate(digits(m)):
            x = self._sk(week, i)
            for j in range(b):
                x = f(x, self.op, week, i, j)
            sig.append(x)
        return sig

    def sign_once(self, m: bytes, week: int, log: "SignLog") -> list[bytes]:

        log.reserve(self.op, week, m)
        return self.sign(m, week)


class KeyReuseError(RuntimeError):
    pass


class SignLog:


    def __init__(self, path):
        self.path = os.path.realpath(os.fspath(path))
        self.bound: dict[tuple[int, int], str] = {}
        if os.path.exists(self.path):
            with self._locked() as fh:
                self.bound = self._read(fh)

    @contextmanager
    def _locked(self):
        with open(self.path, "a+b") as fh:
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + 30.0
                while True:
                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError("could not acquire the signing-log lock") from exc
                        time.sleep(0.01)
                try:
                    yield fh
                finally:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield fh
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read(fh) -> dict:
        fh.seek(0)
        bound = {}
        for line in fh:
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                raise ValueError("incomplete signing-log record; signing is disabled")
            r = json.loads(line)
            key = (r["op"], r["week"])
            if key in bound and bound[key] != r["m"]:
                raise KeyReuseError(f"conflicting signing-log records for operator/week {key}")
            bound[key] = r["m"]
        return bound

    def reserve(self, op: int, week: int, m: bytes) -> None:
        key, mh = (op, week), m.hex()
        with self._locked() as fh:
            self.bound = self._read(fh)
            if key in self.bound:
                if self.bound[key] != mh:
                    raise KeyReuseError(f"operator {op} already signed a different statement in week {week}")
                return
            fh.seek(0, os.SEEK_END)
            fh.write((json.dumps({"op": op, "week": week, "m": mh}) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
            if os.name == "posix":
                parent = os.open(os.path.dirname(self.path), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            self.bound[key] = mh


def verify_pk(m: bytes, sig: list[bytes], op: int, week: int) -> bytes:
    ends = []
    for i, b in enumerate(digits(m)):
        x = sig[i]
        for j in range(b, W1):
            x = f(x, op, week, i, j)
        ends.append(x)
    return k256(*ends)


def auth_path(levels: list[list[bytes]], index: int) -> list[bytes]:
    path = []
    for lvl in range(H):
        path.append(levels[lvl][index ^ 1])
        index >>= 1
    return path


def root_from(leaf: bytes, index: int, path: list[bytes]) -> bytes:
    h = leaf
    for lvl in range(H):
        h = k256(h, path[lvl]) if ((index >> lvl) & 1) == 0 else k256(path[lvl], h)
    return h
