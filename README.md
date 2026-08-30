# HPVSim_Zambia

An [HPVsim](https://hpvsim.org) model of cervical cancer for Zambia, with HIV
co-infection dynamics, calibrated to national HIV, ART, and cancer incidence
data. Built on **hpvsim v2.x** (not yet migrated to v3.x).

## Install

```bash
pip install -r requirements.txt
```

Requires `hpvsim==2.2.6`.

## What's here

| File | Purpose |
|------|---------|
| `run_sim.py` | Defines a standalone single-sim / vaccination-scenario runner. |
| `run_functions.py` | Core simulation, calibration-analysis, and batch-run helpers (used by `run_top_calibrations.py`). |
| `run_calibration.py` | Runs and loads the Optuna-based calibration to HIV/HPV/cancer targets. |
| `run_top_calibrations.py` | Runs simulations across the top-N calibrated parameter sets, with optional age-stratified analyzers and ART-coverage counterfactuals. |
| `data/` | Calibration targets and datafiles (HIV incidence/mortality, ART coverage, cancer incidence). |
| `hpc/` | SLURM job scripts for running calibration/sims on HPC. |
| `tests/` | Smoke tests (baseline sim) and regression tests for `run_functions.py` helpers. |

## Calibration status

A calibration is already committed: `results/zambia_calib.obj` (full calibration
object) and `results/zambia_pars_nov06.obj` (best-fit parameters).

## How to run

Each script has a `to_run` list near its `__main__` block — edit that list to
select which stage to run.

```bash
python run_calibration.py         # calibrate (VM) or load + plot (local); see to_run in the file
python run_sim.py                 # single run / vaccination scenario; see to_run in the file
python run_top_calibrations.py    # simulate the top-N calibrated parameter sets
```

Calibration (`run_calibration.py` with `run_calibration` in `to_run`) and large
batch runs (`run_top_calibrations.py`) are long-running and should be run on a
VM, not locally.

## Testing

```bash
pytest tests/
```

## Data provenance

Sexual debut and partnership parameters are derived by fitting to the 2018
Zambia DHS; see https://www.researchsquare.com/article/rs-3074559/v1.
