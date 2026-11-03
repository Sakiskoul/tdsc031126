# ARGUS: code and CSV experiments

Finite-population monetary-loss models, control-calibrated sequential monitoring, Bitcoin replay evaluation and hash-based alarm authentication.

The repository contains source code, tests, scientific configuration and CSV data. Numerical reference results retain unfavorable findings. Historical analyses are exploratory. Synthetic injections have not been proved covert on the actual ledger, and addresses remain proxies for keys.

## Install

Use Python 3.9.10 for the recorded environment. Create a virtual environment and run:

```text
python -m pip install -r requirements.txt
```

The pinned packages describe the environment used to check this release. They are not a recovered lockfile for every historical extraction host. The EVM benchmark has a separate npm lockfile under `contracts/`.

## Reproduce the principal results

Run from the repository root:

```text
python -m pytest code/tests -q -p no:cacheprovider
python analysis/controlled_validation.py
python analysis/verify_replay.py
python analysis/analyse_saved.py
python analysis/check_joint_and_fees.py
python analysis/screened_power.py
```

These commands respectively check the implementation, simulate known-law sensitivity, reconcile the 12,160 replay cases and 41-point curves, reproduce paired uncertainty and chronological evaluation, reconcile the 5,376 joint-pilot outcomes with assumed fees, and rerun 450 aggregate attacks across the three control cohorts. Generated files are written under `results/generated/`.

The replay and joint-pilot reconciliation commands consume saved CSV trajectories and scores. They do not rebuild the raw blockchain, regenerate all target draws or execute real transactions. The screened-control command performs fresh fitting and aggregate injections. Depending on the host, it can take several minutes.

The initial release check passed 118 tests, with one optional reference-checkout test skipped. The specific NEWMA reference test requires an external checkout selected through `NEWMA_REPO`, at commit `73ebf8f372490c4e4f619ad0bf94f88962291029`. The reference implementation is not bundled.

## Main entry points

| Purpose | Entry point |
|---|---|
| Known-law loss, truncation and assumption sensitivity | `analysis/controlled_validation.py` |
| Full saved replay reconciliation | `analysis/verify_replay.py` |
| Paired median/P90 loss, detection intervals and rolling origins | `analysis/analyse_saved.py` |
| Joint theft/control manipulation and fee accounting | `analysis/check_joint_and_fees.py` |
| Screened controls at identical absolute theft plans | `analysis/screened_power.py` |
| Main aggregate evaluation and historical alarms | `code/experiments/run_all.py` |
| Event-level replay generation | `code/experiments/replay.py` |
| Grouped-control exploratory check | `code/experiments/group_regime.py` |
| Control-variance exploratory check | `code/experiments/control_variance_quick.py` |
| Matched drift-floor diagnostic | `code/experiments/matched_drift_diagnostic.py` |
| Local EVM authentication/gas benchmark | `contracts/measure_gas.js` |

Detailed commands, expected outputs, input schemas and limitations are in `docs/REPRODUCIBILITY.md`. `docs/RESULT_MAP.md` connects reported quantities to their scripts, configurations and CSV inputs. `MANIFEST.csv` identifies the files in the release by SHA-256.

## Scope

The supplied CSV files support inspection and reconstruction of the principal saved results, including all address replay score trajectories. Full raw-ledger extraction and fresh target-pool generation require the public source records and additional storage. Construction and exposure-audit code are included. A complete mechanism-isolating output/address factorial, screened event replay and a script-valid fee-constrained poisoning replay are not completed results of this codebase.

Maintainer: `akoulianos`.
