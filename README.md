# HPVSim_Zambia

An [HPVsim](https://hpvsim.org) model of cervical cancer for Zambia, with HIV
co-infection dynamics, calibrated to national HIV, ART, and cancer incidence
data. Built on **hpvsim v3.2** with **stisim v1.6** (HIV co-infection via
`hpv.Sim(model_hiv='incidence')`).

## Install

```bash
pip install -r requirements.txt
```

Requires `hpvsim[hiv]>=3.2` and `stisim[hiv]>=1.6`.

## What's here

| File | Purpose |
|------|---------|
| `run_scenarios.py` | Runs a single baseline sim, or a baseline-vs-vaccination MultiSim scenario. |
| `run_functions.py` | Core simulation, calibration-analysis, and batch-run helpers (used by `run_top_calibrations.py`). |
| `run_calibration.py` | Runs and loads the Optuna-based calibration to HIV/HPV/cancer targets. |
| `run_top_calibrations.py` | Runs simulations across the top-N calibrated parameter sets, with optional age-stratified analyzers and ART-coverage counterfactuals. |
| `data/` | Calibration targets and datafiles (HIV incidence/mortality, ART coverage, cancer incidence). |
| `hpc/` | SLURM job scripts for running calibration/sims on HPC. |
| `tests/` | Smoke tests (baseline sim) and regression tests for `run_functions.py` helpers. |

## Calibration status

A first-pass v3.2 calibration is committed at `results/zambia_calib.obj`
(top-50 shrunk, ~140 KB). Rerun `run_calibration.py` for a fresh calibration,
or regenerate individual sims from the shrunk artifact via `hpv.make_calib_sims`.
The legacy v2 best-pars file `results/zambia_pars_nov06.obj` is preserved for
reference — see `VM_HANDOFF.md` and `run_functions.v2_calib_pars_to_v3()` for
the v2 → v3 translation.

## How to run

Each script has a `to_run` list near its `__main__` block — edit that list to
select which stage to run.

```bash
python run_calibration.py         # calibrate (VM) or load + plot (local); see to_run in the file
python run_scenarios.py           # single run / vaccination scenario; see to_run in the file
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
