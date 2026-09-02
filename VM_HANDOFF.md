# Handoff: Zambia recalibration + v2.2.6 baseline, on zebra

Written 2026-09-02, after completing the v2.2.6 → v3 port. Supersedes
`ZAMBIA_PORT_HANDOFF.md`, which described the port before it was done and had
the annual-probability conversion backwards.

Everything below was verified against live code and live runs unless marked
**[unverified]**.

## Your mission

Two compute-heavy tasks on `zebra`, in this order:

1. **Recalibrate Zambia under hpvsim v3.** The old calibration is gone (see
   "State of the calibration" below) and the model needs refitting.
2. **Stand up the v2.2.6 baseline and compare it to the v3 port**, to confirm
   nothing was lost in translation and to settle one open modelling question.

Task 2 is diagnostic and independent of task 1; if task 1's calibration is
queued and running, do task 2 while it runs.

## Where things stand

The port is committed as `bde74d4` on branch `v3-port` in this repo (3 commits
ahead of `main`). `pytest tests/` passes 12 tests.

Full-resolution run, 1960–2020, with the old v2 calibrated parameters
translated to v3 names (`rf.v2_calib_pars_to_v3`):

| | modelled | reference |
|---|---|---|
| Population 2020 | 18.9M | 18.4M (UN) |
| ASR cancer incidence | 78.6 | 65.5 (Globocan target) |
| HIV prevalence 15–49 | 0.180 | ~0.11 (UNAIDS) |
| ART coverage | 0.916 | ~0.90 (UNAIDS) |

Population and ART are good. ASR is ~20% high. **HIV prevalence is ~64% high,
and that is the open question for task 2.**

## Setup on zebra

hpvsim is already cloned there in editable form. You need it on the right
branch:

- Branch: **`restore-hpv-latency`**, not `main`. Everything this port depends
  on (HPV latency, unified parameter routing, the `model_hiv=` redesign, and
  three HIV-data fixes made during the port) lives there and is **not merged**.
- Confirm before running anything:
  `python -c "import hpvsim; print(hpvsim.__version__, hpvsim.__file__)"`
  — the path must point at the clone, and the version will read `3.1.0` even
  though `CHANGELOG.md` says 3.2.0 in progress (the version string hasn't been
  bumped yet).
- **Do not `pip install -r requirements.txt`.** It pins `hpvsim[hiv]>=3.2`,
  which won't resolve (3.2 isn't on PyPI) and would clobber the editable
  install. Install the other requirements individually if any are missing.
- `stisim` is required — HIV modelling needs it, and it's an optional hpvsim
  extra as of 3.2. Verify `import stisim` works.

Sizing and hygiene:

- **Get the core count from the machine, not from this document:** `nproc`.
  zebra is expected to have ~160 cores; use all of them as sole user.
- **Check for other users before saturating**: `who`, `w`, and
  `ps -eo user:20,pcpu,pmem,etime,command --sort=-pcpu | head -20`. If someone
  else is on it, surface the conflict rather than competing for cores.
- **Run in `tmux`.** zebra is non-spot so there's no reclamation risk, but the
  calibration is a multi-hour job and a dropped SSH session shouldn't kill it.
- zebra is normally deallocated; starting takes ~2 minutes.

## Task 1 — v3 recalibration

`run_calibration.py`, with `to_run = ['run_calibration']` and `debug = False`
(both already set).

**Raise `n_workers` before you run.** The file hardcodes `n_workers = 50`,
inherited from the v2 setup, which was capped because v2's Optuna sqlite
storage deadlocked past ~32 workers and needed a MySQL server to go higher.
v3's `hpv.Calibration` defaults to Optuna `JournalStorage` specifically to lift
that limit, and the MySQL `storage=` argument has been removed from the script.
So set `n_workers` from `nproc` — on 160 cores you're currently leaving
two-thirds of the machine idle.

**Review the parameter bounds before burning 5000 trials.** They were inherited
from v2 and three things have changed underneath them:

- `beta=[0.05, 0.02, 0.5, 0.02]` — v2's best fit was 0.1; v3's per-genotype
  default is 0.25. The range still covers both, but check the guess.
- The cross-layer bounds have **already** been converted per-timestep → annual
  in the script (`m_cross_layer`, `f_cross_layer`). Don't convert them again.
  Their `step` was dropped deliberately: v2's 0.05 grid doesn't survive the
  non-linear conversion, and an off-grid step makes Optuna silently truncate
  the upper bound.
- **`cross_immunity.rel_sev` is not in the spec but was calibrated in v2**
  (`sev_dist`, fitted to mean 1.33 against a default of 1.0). v3 renamed it and
  moved it onto the `CrossImmunity` connector. Since ASR runs ~20% high, this is
  the most likely single parameter to open up. Note it's a *different*
  parameter from `hiv.rel_sev_lo/_hi`, which are CD4-stratified progression
  multipliers and are already in the spec.

**Targets** are `data/zambia_cancer_cases.csv` (by age, → `all_hpv.cancers.<bin>`)
and `data/zambia_asr_cancer_incidence.csv` (scalar).
`data/zambia_cancer_rate_ratios.csv` is deliberately excluded — see "Known
traps" #3.

**HIV prevalence is not a target.** The HIV parameters are in the spec, but
nothing in `data=` constrains prevalence, so the calibration has no pressure to
fix the 0.180-vs-0.11 gap. Resolve task 2 before deciding whether to add an HIV
prevalence target via a custom `eval_fn` — if the cause is the dropped mortality
data, fitting HIV parameters to compensate would be the wrong fix.

Save with `calib.shrink()` (the script already does). It drops the embedded
sims, leaving an object small enough to commit and — unlike the v2 artefact —
loadable without the original environment.

## Task 2 — v2.2.6 baseline comparison

The goal is one specific question plus a general check.

**The specific question: does dropping imposed HIV mortality explain the
prevalence overshoot?** v2 fed Zambia's HIV mortality in directly from
`data/zambia_female_hiv_mortality_updated.csv` and
`data/zambia_male_hiv_mortality_updated.csv`. v3 has no mortality input at all —
stisim derives it endogenously from CD4 progression. If v3's endogenous
mortality is lower than Zambia's imposed rates, prevalence accumulates, which
would explain 0.180 against ~0.11. **This is the leading hypothesis but is
[unverified]** — it could equally be the incidence curve being applied
differently (v3 converts the annual rate with `p = 1 - exp(-r·dt)` over
susceptibles). Compare HIV deaths between the two versions to separate these.

If imposed mortality turns out to matter, that's a real modelling decision for
Robyn, not a mechanical fix — v3 has no hook to impose it, so the options are
calibrating stisim's CD4/mortality parameters or adding an intervention.

**Getting the v2 code and environment:**

- v2.2.6 needs its own environment. On Robyn's laptop it lives in the `starsim`
  conda env (hpvsim 2.2.6 from site-packages); on zebra, create a fresh one — do
  not install 2.2.6 anywhere that shadows the v3 editable clone.
- The pre-port code is the parent of the port commit: `git show bde74d4~1:run_sim.py`,
  and likewise `run_functions.py`, `run_calibration.py`. Cleanest is a separate
  worktree at `bde74d4~1` so both versions exist side by side.
- All the data files v2 needs are still in `data/`, untouched — including the two
  mortality CSVs the v3 port no longer reads.

**Compare, at 2020:** population, ASR cancer incidence, HIV prevalence 15–49,
ART coverage, `cancers_with_hiv` / `cancers_no_hiv`, and HIV deaths. Use the same
`rand_seed` and enough seeds to distinguish signal from noise.

**When reading the v2 numbers**, note the result names moved: v2's flat
`sim.results['cancers']` is v3's `sim.results.all_hpv.new_cancers`, and the
HIV-stratified cancer results are on `all_hpv`, not `sim.results.hiv`.

## Known traps

1. **`sim.results.n_alive` is wrong at high `ms_agent_ratio`.** It ignores
   per-agent multiscale weights and over-reports by ~3.4× at Zambia's
   `ms_agent_ratio=100` (1.00× at ratio 1, 1.31× at 100, compounding over a
   run). For population, use
   `people.scale.values[people.alive.values].sum() * sim.pars.pop_scale`.
   `asr_cancer_incidence` is unaffected — its denominator is scale-weighted.
   This is an upstream reporting bug, not a dynamics bug; the model's population
   is correct.
2. **`all_hpv.cancer_rate_ratio` silently returns 0.0** when a timestep has no
   HIV-negative cancers, which is common at small agent counts. Don't use it as
   a calibration target without pooling over a multi-year window first.
3. **By-age HIV-stratified targets can't go through `data=`.** The file loads
   fine, but `Calibration` routes age-stratified targets through an
   auto-created `by_age` analyzer whose key whitelist has no HIV-stratified
   entries, so it raises `by_age: unknown key(s) ['cancer_hiv_rate_ratios']`.
   Use `analyzers.CancerByAgeHIV` (Layer 3, in this repo) with a custom
   `eval_fn`; hpvsim's `tests/regression/calibrate_rwanda.py` is the
   hand-rolled-objective precedent. Note also that the v2 script pointed at
   `data/cancer_rate_ratios.csv`, without the `zambia_` prefix — that file
   doesn't exist, so this target was never actually being fitted.
4. **The v2 `.obj` files can't be unpickled under v3.** They reference
   `hpvsim.people`, `hpvsim.analysis` and `hpvsim.base`, all deleted; sciris
   returns a `NamedFailed` placeholder rather than raising.
   `results/v2_artefact_snapshot.json` preserves their contents as JSON.
5. **An `ss.Dist` carries RNG state and must not be shared between sims** — a
   module-level distribution constant raises `DistSeedRepeatError` on the second
   sim. `rf._behaviour_dists()` builds fresh ones per call; follow that pattern
   if you add parameters.

## State of the calibration

`results/zambia_calib.obj` is **a single-trial debug run**, not a real
calibration — 1 row, mismatch 9.95, with `hpv_control_prob=1.0` and
`hpv_reactivation=0.0` sitting at their bound corners. There is no top-100
ensemble and there never was, so `run_top_calibrations.py` (which asks for
`n=100`) would have silently run 1 parameter set.

`results/zambia_pars_nov06.obj` is a plain dict and does load, but is mostly v3
defaults. The genuinely-fitted values in it:

| parameter | v2 value | v3 default |
|---|---|---|
| `beta` | 0.1 | 0.25 |
| `m_cross_layer` / `f_cross_layer` | 0.35 / 0.1 (per-timestep) | 0.76 / 0.185 (annual) |
| `m_partners_casual` / `f_partners_casual` | 0.34 / 0.26 | 0.5 / 0.5 |
| `cin_fn.k` (hpv16/18/hi5/ohr) | 0.35 / 0.4 / 0.35 / 0.28 | 0.3 / 0.25 / 0.2 / 0.2 |
| `cancer_fn.k` (hpv16/hi5/ohr) | 0.25 / 0.15 / 0.15 | 0.3 / 0.2 / 0.2 |
| `sev_dist` → `cross_immunity.rel_sev` | 1.33 | 1.0 |
| `hiv.rel_reactivation_lo/_hi` | 3 (single v2 value, now split) | 1.0 / 1.0 |

`rf.v2_calib_pars_to_v3()` performs this translation, including the annual
conversion. Use it as the starting point / sanity anchor for the new
calibration, not as a finished fit.

## Don't

- Don't `pip install -r requirements.txt` over the editable hpvsim clone.
- Don't commit calibration outputs or simulation results to git. Consolidate on
  the VM, then `rsync` the merged file off.
- Don't run `run_top_calibrations.py` until a real v3 calibration exists — it's
  100 parameter sets × 100 runs = 10,000 sims and its input is currently the
  1-trial debug object.
- Don't deallocate zebra without checking `who` and for detached tmux sessions.
- Don't edit files in `data/` — including the two HIV mortality CSVs, which look
  unused but are needed for task 2.
- Don't commit anything in the hpvsim clone if you work there: that tree may also
  hold another agent's in-progress docs work for the v3.2 PR.

## The annual-probability conversion, stated correctly

`layer_probs` and the cross-layer probabilities were reinterpreted from
per-timestep to **annual** in v2.3.0. Zambia's were fitted under v2.2.6, so they
need converting:

```python
p_annual = 1 - (1 - p_per_timestep) ** (1 / dt)
```

**The exponent is `1/dt`, not `dt`.** At `dt=0.25`, `p=0.1` → `0.344`. Using
`**dt` gives `0.026` — the wrong direction, and it causes exactly the silent
transmission collapse the conversion exists to prevent. The old handoff doc had
this backwards. `rf._to_annual_prob` implements it correctly and is already
applied to both the hardcoded behaviour parameters and the calibration bounds.
