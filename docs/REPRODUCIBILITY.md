# Reproduction guide

## Inputs and numerical representation

Only CSV data are distributed. `code/experiments/data_io.py` reads the CSV counterparts of the original table paths, reassembles chunked tables and restores the scientific column types recorded in `config/csv_schemas.csv`. It also rebuilds the in-memory replay tensors from CSV. It can consume locally regenerated Parquet/NPZ inputs when supplied, but those formats are not part of this release.

CSV floating-point values are written with 17 significant digits and read with round-trip parsing. Stored weekly replay scores and stolen amounts are restored to their original float32 type. Nominal losses and paired nominal intervals use the saved double-precision outcomes. The maximum discrepancy when curve losses are reconstructed from float32 weekly amounts is 0.379273 BTC.

The CSV conversion was checked against 142 original tables with exact DataFrame equality, including types and dates. Anonymous byte identifiers are represented as hexadecimal strings in CSV and restored by the loader. They are ledger-derived digests, not private keys.

| Data location | Meaning and units |
|---|---|
| `data/results_full/` | Sparse output risk/event tables. `sv` is satoshis and `sv2` is BTC squared. |
| `data/results_addr/` | Corresponding address-proxy tables with the same units. |
| `data/screened/` | Dense screened panels after unit conversion. `sv` and `e_sv` are BTC, second moments are BTC squared. |
| `data/event_replay/runs/` | One CSV per start. Columns identify draw, law, intensity, week, weekly stolen BTC, total selected proxies and nine detector scores. |
| `data/event_replay/nominal_outcomes.csv` | One row per attack/detector, retaining double-precision loss, alarm, delay and draw-zero frozen outcomes. |
| `data/event_replay/natural_weekly.csv` | The corresponding untouched weekly scores. These transactions are not independently certified benign. |
| `data/control_pilot/plans/` | Shared theft and control plans. `btc` is BTC, `initial_sat` and `moved_sat` are satoshis. |
| `data/control_pilot/history_*.csv` | The pre-onset score histories used by the paired joint analysis. |
| `data/control_pilot/starts/` | Saved joint-pilot score/flow curves, one CSV per start. |
| `results/` | Reference weekly series and CSV summaries. |

Weeks start on Monday in UTC. The exclusive ledger cutoff is 2026-10-01. Initial hazard fitting uses 2014-01-06 inclusive to 2019-12-30 exclusive, while calibration includes the 2019-12-30 week. Evaluation spans 2020-01-06 through 2026-09-28. Rolling refits use a 104-week lookback, a 13-week gap and a global 13-week cadence. Strata, fitting updates, prior smoothing, variance inflation and seeds are specified in `config/configuration.json`.

## Saved replay, uncertainty and chronological analysis

```text
python analysis/verify_replay.py
python analysis/analyse_saved.py
```

The first command independently recalculates all saved nominal rates/median losses and all 41 threshold curves, checks the complete 76 x 8 x 4 x 5 grid, spending caps, loss accounting and detector-specific crossing inequalities. Recorded no-attack reconstruction residuals and quota certificates are checked as saved records. This is not a new reconstruction from raw blocks.

The second command produces paired excess-detection and median/P90-loss differences for tilted unit-intensity attacks. It resamples 13-start chronological blocks and retains all eight target draws within each start. Temporal and target-draw variability are distinct. Rolling origins use only calibration windows ending before the next evaluation onset. Four origins cover 50 evaluation starts. History had already been inspected, so this is retrospective exploration rather than a new confirmatory holdout. Thresholds that do not attain a budget on the fixed grid remain unavailable.

## Controlled-law checks

```text
python analysis/controlled_validation.py
```

This uses 50,000 replicates per setting, exact homogeneous-binomial TV and a small exact truncation calculation. The Bernoulli and Poisson cases are distinct models. Batch dependence, baseline error and drift are deliberate violations of the oracle assumptions. Their results do not certify Bitcoin calibration.

## Joint manipulation of the control

```text
python analysis/check_joint_and_fees.py
python code/control/analyse_joint.py
```

The first command reconstructs detector decisions from the CSV histories, calibrated parameters and saved joint curves, then reconciles all 5,376 losses to the shared theft plans. Fee calculations assume a P2WPKH consolidation transaction per activated address, 68 vbytes/input, 31 for change, 10.5 overhead and rates of 1, 10 or 100 sat/vbyte. These are illustrative fees, not actual signed transactions or a new fee-constrained replay.

The second command reproduces the original paired pilot summaries from the CSV outcomes. The capped control uses a 0.1 BTC influence cap per address, with the working moment corrections implemented in `code/control/control_common.py` and `fast_stats.py`. Capital charges all initial linked inventory, including initially immature holdings. Eight historical starts and one target draw limit generalisation. The available bucket-zero inventory is 79.7--228.3 BTC, so a 1,000 BTC ceiling cannot be fully exercised. Coin age is not ownership duration.

`code/control/joint_replay.py` contains the simulation logic. Fresh theft-plan generation also needs the onset target-pool exports created from the raw address ledger. Those full exports are not included. The supplied curves and shared plans support complete reconciliation of the reported pilot, not arbitrary new target draws.

## Screened controls

```text
python analysis/screened_power.py
```

This refits and calibrates each cohort using the same training protocol and reruns 75 starts at each of two intensities, reusing identical absolute BTC/count plans. The exposed rows are checked to remain unchanged. It produces 450 aggregate attacks. Screened event-level replay is not implemented by this command.

## Aggregate evaluation and drift

```text
python code/experiments/run_all.py --results data/results_full --t-end 2026-10-01 --train 2014-01-06 2019-12-30 --eval 2020-01-06 2026-09-28 --tag full
python code/experiments/run_all.py --results data/results_addr --t-end 2026-10-01 --train 2014-01-06 2019-12-30 --eval 2020-01-06 2026-09-28 --tag addr
python code/experiments/matched_drift_diagnostic.py --out results/generated/drift
python code/experiments/rq4_analysis.py --tag full
python code/experiments/rq7_pools.py --tornado results/rq7_weekly_full.csv --tag reproduced
```

These reproduce fitted summaries and aggregate injections from the supplied weekly cohorts. The nominal training run-length target does not imply the same realised window alarm rate for every method. Delay is conditional on an alarm, whereas loss includes the alarm week or the whole horizon. Equal normalised intensity across panels need not mean equal absolute BTC.

## Exploratory negative controls

Prepare output directories and copy the scientific protocols with this platform-independent command:

```text
python -c "from pathlib import Path; import shutil; pairs=[('grouped','grouped_control.json'),('variance','control_variance.json')]; [(Path('results/generated',n).mkdir(parents=True,exist_ok=True),shutil.copyfile(Path('config',c),Path('results/generated',n,'protocol.json'))) for n,c in pairs]"
python code/experiments/group_regime.py --panel full --out results/generated/grouped --evaluate
python code/experiments/group_regime.py --panel addr --out results/generated/grouped --evaluate
python code/experiments/control_variance_quick.py --panel full --out results/generated/variance
python code/experiments/control_variance_quick.py --panel addr --out results/generated/variance
```

Grouping selection is based on 2019 following a 2014--2018 fit. Failed selection gates are retained. Both investigations followed inspection of the evaluation history. The published script hash in the variance configuration binds the distributed source, not a claim of a new historical preregistration.

## Ledger construction and fresh replay

The `code/data/` scripts document the public source schema, partitioned joins, exposure intervals, recognised key derivations and target export. Source data are read anonymously from the public blockchain dataset. Extraction is explicit and may require substantial storage and time. An immutable raw-provider snapshot identifier and canonical chain-tip hash were not recovered. The CSV manifest identifies the derived release inputs.

After separately constructing the full address ledger and exporting the target pools, the `replay.py prep`, `run` and `curves` commands regenerate the event design. Use `python code/experiments/replay.py --help` and the subcommand help for the required locations. Preserve the global refit calendar, onset-only ranking, whole-address thefts, cancellation from the theft week and risk removal from the following week.

## Local EVM benchmark

With Node.js and npm installed:

```text
cd contracts
npm ci
node measure_gas.js
```

Set `PYTHON` to the interpreter from the Python environment when `python` resolves elsewhere. The benchmark runs Ganache in process and creates deterministic synthetic operator keys. The on-chain prototype verifies Winternitz/Merkle signatures. It does not verify SLH-DSA on chain. Agreement of authenticated operators does not establish statistical correctness, and local sign-once persistence does not protect against storage rollback.

## Unfinished experiments

The complete paired decomposition of representation, cancellation, risk moments and refitting remains unresolved. The draw-zero frozen/refitted contrast covers refitting within its declared setting. There is no completed screened event replay, actual-script fee-constrained poisoning evaluation, general poisoning-resistance guarantee or practical superiority claim at all alarm budgets.
