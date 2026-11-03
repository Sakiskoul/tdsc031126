from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / 'data/event_replay'
NAMES = ['argus', 'uncalibrated', 'count', 'eprocess', 'focus', 'newma', 'ctm', 'ecod', 'threshold']
LAWS = ['tilted', 'mimic', 'uniform', 'whale']
INTENSITIES = [0.25, 0.5, 1.0, 2.0, 4.0]


def read_csv(path, **kwargs):
    kwargs.setdefault('float_precision', 'round_trip')
    return pd.read_csv(path, **kwargs)


@lru_cache(maxsize=1)
def csv_schema():
    return read_csv(ROOT / 'config/csv_schemas.csv')


def read_frame(path, **kwargs):
    path = Path(path)
    if path.exists():
        if path.suffix == '.csv':
            frame = read_csv(path, **kwargs)
        else:
            return pd.read_parquet(path, **kwargs)
    elif path.with_suffix('.csv').exists():
        frame = read_csv(path.with_suffix('.csv'), **kwargs)
    else:
        files = sorted(path.with_suffix('').glob('part_*.csv'))
        if not files:
            raise FileNotFoundError(f'Required input is missing: {path.name}')
        frame = pd.concat([read_csv(p, **kwargs) for p in files], ignore_index=True)
    schema_path = ROOT / 'config/csv_schemas.csv'
    if schema_path.exists():
        schema = csv_schema()
        try:
            relative = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            relative = ''
        for row in schema[schema.path == relative].itertuples(index=False):
            if row.kind == 'datetime':
                frame[row.column] = pd.to_datetime(frame[row.column]).astype(row.dtype)
            elif row.kind == 'date':
                frame[row.column] = pd.to_datetime(frame[row.column]).dt.date
            elif row.kind in ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']:
                frame[row.column] = frame[row.column].astype(row.kind)
            elif row.kind == 'bytes':
                frame[row.column] = frame[row.column].map(lambda value: bytes.fromhex(value) if isinstance(value, str) else None)
    return frame


class ReplayArrays(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@lru_cache(maxsize=1)
def natural_weekly():
    return read_csv(REPLAY / 'natural_weekly.csv')


def replay_arrays(path, **kwargs):
    path = Path(path)
    if path.exists() and path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as stored:
            return ReplayArrays({k: stored[k] for k in stored.files})
    frame = read_csv(path.with_suffix('.csv'))
    sid = int(path.stem.split('_')[-1])
    assert len(frame) == 8 * 4 * 5 * 52
    scores = frame[NAMES].to_numpy(dtype=np.float32).reshape(8, 4, 5, 52, 9).transpose(0, 1, 2, 4, 3)
    base = natural_weekly()
    base = base[base.sid == sid].sort_values('week')[NAMES].to_numpy(dtype=np.float32).T
    return ReplayArrays(names=np.array(NAMES), draws=np.arange(8), laws=np.array(LAWS),
                        intensities=np.array(INTENSITIES), scores=scores, base=base,
                        stolen=frame.stolen_btc.to_numpy(dtype=np.float32).reshape(8, 4, 5, 52),
                        keys=frame['keys'].to_numpy(dtype=np.int64).reshape(8, 4, 5, 52)[..., 0])


@lru_cache(maxsize=1)
def nominal_outcomes():
    return read_csv(REPLAY / 'nominal_outcomes.csv')


@lru_cache(maxsize=76)
def read_checkpoint(path):
    path = Path(path)
    if path.suffix == '.json' and path.exists():
        return json.loads(path.read_text())
    sid = int(Path(path).stem.split('_')[-1])
    metadata = read_csv(REPLAY / 'starts.csv').set_index('sid').loc[sid]
    base = natural_weekly()
    base = base[base.sid == sid]
    natural = {name: int(((base[name] >= 1) if name in ['eprocess', 'ctm', 'threshold'] else (base[name] > 1)).any()) for name in NAMES}
    outcomes = nominal_outcomes()
    outcomes = outcomes[outcomes.sid == sid]
    runs = []
    scalar = ['draw', 'law', 'm', 'planned_btc', 'stolen_btc', 'keys', 'keys_week_max', 'pool_size', 'unmatched_delta_share']
    for _, group in outcomes.groupby(['draw', 'law', 'm'], sort=False):
        record = {k: group.iloc[0][k] for k in scalar}
        record['draw'] = int(record['draw'])
        record['det'] = {}
        frozen_alarm, frozen_delay = {}, {}
        for row in group.itertuples(index=False):
            record['det'][row.detector] = {'alarm': int(row.alarm), 'delay': None if pd.isna(row.delay) else int(row.delay), 'loss_btc': float(row.loss_btc)}
            if pd.notna(row.frozen_alarm):
                frozen_alarm[row.detector] = int(row.frozen_alarm)
                frozen_delay[row.detector] = None if pd.isna(row.frozen_delay) else int(row.frozen_delay)
        if frozen_alarm:
            record['frozen_alarm'] = frozen_alarm
            record['frozen_delay'] = frozen_delay
        runs.append(record)
    return {'sid': sid, 't0': metadata.t0, 'check_rel': float(metadata.check_rel),
            'pool': {'bucket_quota_verified': bool(metadata.bucket_quota_verified)}, 'natural': natural, 'runs': runs}


@lru_cache(maxsize=1)
def replay_report():
    starts = [read_checkpoint(REPLAY / 'runs' / f'start_{sid:03d}.csv') for sid in range(76)]
    summary = {'starts': 76, 'natural_window_alarm_rate': {name: np.mean([s['natural'][name] for s in starts]) for name in NAMES}, 'laws': {law: {} for law in LAWS}}
    for row in read_csv(REPLAY / 'law_summary.csv').to_dict('records'):
        law, m = row.pop('law'), row.pop('m')
        summary['laws'][law][str(float(m))] = dict(row, det={})
    for row in read_csv(REPLAY / 'summary.csv').to_dict('records'):
        law, m, name = row.pop('law'), row.pop('m'), row.pop('detector')
        row['excess_ci95'] = [row.pop('excess_ci_low'), row.pop('excess_ci_high')]
        summary['laws'][law][str(float(m))]['det'][name] = row
    return {'summary': summary, 'per_start': starts}


def replay_curves():
    frame = read_csv(REPLAY / 'operating_curves.csv')
    grid = sorted(frame.threshold_multiplier.unique())
    detectors = {}
    for name in NAMES:
        selected = frame[frame.detector == name]
        natural = selected[(selected.law == 'tilted') & (selected.m == 1)].sort_values('threshold_multiplier')
        record = {'natural_rate': natural.natural_rate.tolist(), 'laws': {law: {} for law in LAWS}}
        for (law, m), group in selected.groupby(['law', 'm'], sort=False):
            group = group.sort_values('threshold_multiplier')
            record['laws'][law][str(float(m))] = {'detection': group.detection.tolist(), 'median_loss_btc': group.median_loss_btc.tolist()}
        detectors[name] = record
    return {'starts': 76, 'draws': 8, 'kappas': grid, 'detectors': detectors}


def joint_curves(path):
    frame = read_csv(Path(path).with_suffix('.csv'))
    return {column: frame[column].to_numpy() for column in frame if column != 'week'}


def joint_checkpoint(sid):
    frame = read_csv(ROOT / 'data/control_pilot/all_outcomes.csv')
    return {'outcomes': frame[frame.sid == sid].to_dict('records')}
