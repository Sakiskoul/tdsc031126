import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus import attest as A

OP = A.Operator(op=3, seed=b"argus-test-operator-3", start_week=2800)


def _stmt(level: int, week: int = 2805) -> bytes:
    return A.statement(1337, bytes(20), week, level, bytes(32))


def test_signature_verifies_against_week_key():
    m = _stmt(2)
    sig = OP.sign(m, 2805)
    assert A.verify_pk(m, sig, OP.op, 2805) == OP.pk(2805)
    assert A.verify_pk(_stmt(1), sig, OP.op, 2805) != OP.pk(2805)


def test_sign_once_is_idempotent_and_refuses_a_second_statement(tmp_path):
    log = A.SignLog(tmp_path / "signlog.jsonl")
    s1 = OP.sign_once(_stmt(2), 2805, log)
    assert OP.sign_once(_stmt(2), 2805, log) == s1
    try:
        OP.sign_once(_stmt(3), 2805, log)
        refused = False
    except A.KeyReuseError:
        refused = True
    assert refused
    OP.sign_once(_stmt(3, week=2806), 2806, log)


def test_sign_log_survives_a_restart(tmp_path):
    path = tmp_path / "signlog.jsonl"
    OP.sign_once(_stmt(1), 2805, A.SignLog(path))
    restarted = A.SignLog(path)
    try:
        OP.sign_once(_stmt(2), 2805, restarted)
        refused = False
    except A.KeyReuseError:
        refused = True
    assert refused


def test_sign_log_refreshes_preexisting_instances(tmp_path):
    import pytest

    path = tmp_path / "signlog.jsonl"
    first, stale = A.SignLog(path), A.SignLog(path)
    first.reserve(OP.op, 2805, _stmt(1))
    with pytest.raises(A.KeyReuseError):
        stale.reserve(OP.op, 2805, _stmt(2))
    stale.reserve(OP.op, 2805, _stmt(1))
    assert len(path.read_text().splitlines()) == 1


def _concurrent_reserve(path, level, barrier, outcomes):
    log = A.SignLog(path)
    barrier.wait(timeout=20)
    try:
        log.reserve(OP.op, 2805, _stmt(level))
        outcomes.put("reserved")
    except A.KeyReuseError:
        outcomes.put("refused")


def test_sign_log_serialises_concurrent_processes(tmp_path):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier, outcomes = ctx.Barrier(4), ctx.Queue()
    path = tmp_path / "signlog.jsonl"
    jobs = [ctx.Process(target=_concurrent_reserve, args=(str(path), level, barrier, outcomes))
            for level in range(1, 5)]
    try:
        for job in jobs:
            job.start()
        result = [outcomes.get(timeout=30) for _ in jobs]
        for job in jobs:
            job.join(timeout=15)
            assert job.exitcode == 0
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
                job.join(timeout=5)
    assert sorted(result) == ["refused", "refused", "refused", "reserved"]
    assert len(path.read_text().splitlines()) == 1


def test_sign_log_fails_closed_on_conflicting_or_incomplete_records(tmp_path):
    import json
    import pytest

    path = tmp_path / "signlog.jsonl"
    records = [{"op": OP.op, "week": 2805, "m": _stmt(level).hex()} for level in (1, 2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(A.KeyReuseError):
        A.SignLog(path).reserve(OP.op, 2805, _stmt(2))
    path.write_text(json.dumps(records[0])[:-4])
    with pytest.raises(ValueError):
        A.SignLog(path).reserve(OP.op, 2805, _stmt(2))


def test_two_statements_under_one_key_expose_lower_chain_values():


    m1, m2 = _stmt(1), _stmt(2)
    d1, d2 = A.digits(m1), A.digits(m2)
    assert any(a != b for a, b in zip(d1, d2))
    s1, s2 = OP.sign(m1, 2805), OP.sign(m2, 2805)
    low = [s1[i] if d1[i] <= d2[i] else s2[i] for i in range(A.LEN)]
    lowd = [min(a, b) for a, b in zip(d1, d2)]

    for i in range(A.LEN):
        x = low[i]
        for j in range(lowd[i], max(d1[i], d2[i])):
            x = A.f(x, OP.op, 2805, i, j)
        assert x == (s1[i] if d1[i] >= d2[i] else s2[i])
