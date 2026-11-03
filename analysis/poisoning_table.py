from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'results/generated'
SRC = ROOT / 'data/control_pilot'
HERE.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SRC / 'all_outcomes.csv', float_precision='round_trip')
inventory = pd.read_csv(SRC / 'inventory_accounting.csv', float_precision='round_trip')
select = df[(df['mode'] == 'refitted') & (df.detector == 'argus') & (df.operating_point == '0.25')]
starts = sorted(select.sid.unique())
rng = np.random.default_rng(2026110310)
idx = (rng.integers(0, 8, size=(10000, 4, 1)) + np.arange(2)).reshape(10000, 8) % 8
rows = []
for method in ['count', 'cap_0_1']:
    b = select[(select.method == method) & (select.budget_btc == 0)]
    for (budget, policy) in [(0, 'none'), (100, 'uniform')]:
        q = select[(select.method == method) & (select.budget_btc == budget) & (select.policy == policy)]
        paired = q.merge(b[['sid', 'law', 'intensity', 'loss_btc']], on=['sid', 'law', 'intensity'], suffixes=('', '_base'), validate='one_to_one')
        paired['difference'] = paired.loss_btc - paired.loss_btc_base
        per = paired.groupby('sid').difference.mean().reindex(starts).values
        (lo, hi) = np.quantile(per[idx].mean(axis=1), [0.025, 0.975])
        cap = inventory[(inventory.budget_btc == budget) & (inventory.policy == policy)]
        rows.append(dict(control=method, budget_btc=budget, policy=policy, scenarios=len(q), mean_baseline_btc=b.loss_btc.mean(), mean_loss_btc=q.loss_btc.mean(), paired_mean_difference_btc=per.mean(), ci_low_btc=lo, ci_high_btc=hi, capital_mean_btc=cap.charged_btc.mean(), capital_min_btc=cap.charged_btc.min(), capital_max_btc=cap.charged_btc.max(), natural_window_rate=q.natural_window_far.iloc[0], threshold_multiplier=q.kappa.iloc[0], detection_rate=q.alarm.mean()))
pd.DataFrame(rows).to_csv(HERE / 'poisoning_table.csv', index=False, float_format='%.17g')
(HERE / 'poisoning_table_protocol.json').write_text(json.dumps({'seed': 2026110310, 'replicates': 10000, 'temporal_units': 8, 'block_length': 2, 'outcome': 'Mean loss, pairing four theft scenarios within each start', 'scope': 'Exploratory eight-start pilot on previously inspected history, one target draw'}, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
