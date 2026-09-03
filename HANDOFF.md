# Handoff: finishing the Zambia v3 port

Written 2026-09-03. Supersedes `VM_HANDOFF.md`, whose two tasks are both done:
the v3 recalibration exists (1000 trials, 50 sets, mismatch 3.44–5.20) and the
hpvsim 2.2.6 baseline comparison has been run (results below).

Five things remain. (1) is the only one with real scientific risk; (2)–(5) are
mechanical but order-dependent.

## Read this first: the figures in `figures/` are meaningless

They do **not** disagree with the manuscript — they were never a real run. The
committed PNGs came from a plumbing test: **1 parameter set, 1 seed,
`debug=1`** (1,000 agents, `dt=1.0`, start 1980). Crude cancer incidence in
2020 reads **0.6 per 100k** against the manuscript's ~22, because a
1,000-agent sim starting in 1980 cannot produce a cervical cancer epidemic.

`results/zambia_figure_*.csv` is from the same test — check `par_set` in it
before believing anything downstream. Only `supplementary_figure1_posteriors.png`
used the real 50-set calibration and is worth looking at.

So do not start by hunting for a bug. Start by producing a real run.

## 1. Sanity-check the analysis against the manuscript

The bar Robyn set: **parameter values will have changed and that is fine, but
the outputs, results and narrative should not move much.** Two reasons the
parameters must change, both already established — do not treat either as a
defect to fix:

- hpvsim 2.2.6's multiscale engine scaled cancer down ~3.2× at
  `ms_agent_ratio=100`, which is what Zambia uses. v2's own ASR goes from 96 to
  309 when you set `ms_agent_ratio=1`; v3 gives ~334 either way. v3 is the
  unbiased one, so the v2 fit cannot carry over.
- v3 changed defaults deliberately (`transm2f` 3.69 → 2.0 and others; see the
  3.1.0 *Regression information* entries).

### Produce the run

```bash
python run_counterfactuals.py --run-sim --n-pars 100 --n-seeds 100   # VM
python plot_figures.py                                              # local
```

`--n-pars 100` needs a calibration holding ≥100 sets; the current one holds 50
because `run_calibration.py` calls `shrink(n_results=50)`. Either raise that
and recalibrate, or run at 50 and say so. **The manuscript uses two different
ensembles** — 200 sets for Figure 1, Table 1 and Supp S1; 100 sets × 100 seeds
for Figures 2–3. That is deliberate, not a typo, so match whichever figure you
are checking.

Note `run_counterfactuals.py` defaults to `--stop 2026` so that 2025, the last
reported year, is a complete calendar year. Annualised results read ~4× low in
a partly covered final year at `dt=0.25`. This has bitten twice already.

### Acceptance criteria, from the manuscript

| Quantity | Manuscript |
|---|---|
| ASR cancer incidence, 2020 | 65.5 per 100k (GLOBOCAN target) |
| Cancer IRR, women with vs without HIV | ~6-fold |
| Crude incidence, women with HIV, 2020 | ~180 per 100k |
| Crude incidence, women without HIV, 2020 | ~12 per 100k |
| Age-specific incidence, WWH aged 45–50, 2025 | 293 per 100k, ~6× women without HIV |
| Peak incidence age, WWH vs without | 45–50 vs 60–65 (~10 years earlier) |
| Cumulative cases by 2025, status quo | 71,042 |
| Cumulative cases by 2025, no HIV | 35,580 |
| HIV-attributable fraction by 2025 | 49.9% |
| Peak HIV-attributable fraction | ~65%, in 2012 |
| Cases averted by ART by 2025 | ≥4,068 overall (5.4%); ≥3,932 among WWH (8.7%) |
| Scenario divergence begins | early 1990s |

The narrative claims matter more than the digits: HIV roughly doubles the
burden, ART's effect appears only in the 2010s and is modest, and WWH peak
about a decade earlier. If those hold, the port is good. `run_counterfactuals.py`
saves `new_cancers`, so the cumulative totals and attributable fractions are
computable from the CSVs without another run.

### What is already known to match, and what is not

A head-to-head against 2.2.6 has been run, same parameters, same seed, 2020:

| | v3 | v2.2.6 | ratio |
|---|---|---|---|
| Population | 19.39M | 18.25M | 1.06× |
| HIV infections | 1.114M | 1.143M | 0.97× |
| HIV prevalence (all-age) | 0.0591 | 0.0626 | 0.94× |
| ART coverage | 0.860 | 0.833 | 1.03× |
| ASR cancer incidence | 334.1 | 96.2 | 3.47× |

HIV reproduces 2.2.6 closely. The cancer ratio is the multiscale bias above,
not a port error. To re-run that comparison: `git worktree add --detach <dir>
35c647d` gives the pre-port code, and hpvsim 2.2.6 lives in the `starsim` conda
env. Use `run_sim(calib_pars=sc.loadobj('results/zambia_pars_nov06.obj'))`.

### If the numbers are off, look here first

- **Calibration bounds are truncating.** In `supplementary_figure1_posteriors.png`,
  `m_cross_layer`, `f_cross_layer`, `hiv.rel_sus_lo`, `hiv.rel_sev_lo` and
  `hiv.rel_reactivation_hi` all pile up against their limits — the fit wants to
  go further than allowed. Widen them and recalibrate before concluding the
  model cannot reach the targets.
- **`hpv_control_prob` and `hpv_reactivation` are unidentified** — all 50 sets
  share one value. The paper did not calibrate latency; consider closing them.
- **`cross_immunity.rel_sev` is not in the calibration spec** but John fitted it
  in v2 (`sev_dist`, 1.33 against a default of 1.0). If ASR runs high, this is
  the single most likely parameter to open.
- **The by-age rate-ratio target is not being fitted.** `hpv.Calibration(data=)`
  cannot express an HIV-stratified by-age target; `run_calibration.py` explains
  why and points at the hand-rolled-objective pattern. Note the v2 code never
  fitted it either — it pointed at `data/cancer_rate_ratios.csv`, without the
  `zambia_` prefix, which does not exist.

## 2. READMEs and PR readiness

`README.md` is 61 lines and predates the port. It needs: the `hpvsim[hiv]>=3.2`
requirement, the new `run_counterfactuals.py` → `plot_figures.py` split, the
`run_sim.py` → `run_scenarios.py` rename, `analyzers.py`, and the fact that
`v2_calib_pars_to_v3()` must be called once per sim (it returns live
`ss.Dist` objects).

Also worth doing:
- Delete this file once its contents have landed. A stale handoff is worse than
  none — the first one had the annual-probability conversion backwards.
- `uplifter_report.md` and `.uplifter_cache/` are untracked leftovers from an
  engineering-quality pass. Read the report for pre-existing issues, then bin them.
- `hpc/*.sh` still point at `/storage/homefs/ja22x644/` and Python 3.10 on a Bern
  cluster. Either update them or drop them.
- `results/zambia_pars_nov06.obj` and the v2 `zambia_calib.obj` cannot be
  unpickled under v3. `results/v2_artefact_snapshot.json` preserves their
  contents; decide whether the `.obj` files stay.
- CI runs `pip install -r requirements.txt` against `hpvsim[hiv]>=3.2`, so it
  stays red until (4) lands.

## 3. Release stisim 1.6.1

Do this first — hpvsim 3.2 pins `stisim>=1.6.1` and cannot release without it.

Branch `rc1.6.1` at `83fcba0` (PR #592 merged). It carries the change hpvsim
depends on: `age_bins=None, sex_keys=None` on any `BaseSTI` subclass suppresses
the age/sex-stratified results. `hpv.HIV` passes both.

Note the `hpvsim` conda env currently has stisim installed **editable** from
`/Users/robynstuart/gf/stisim`, so it follows whatever branch that checkout is
on. Reinstall from PyPI once released, or the VM and laptop will silently
disagree.

## 4. Release hpvsim 3.2

Branch `restore-hpv-latency` at `05386959`, version already `3.2.0`, 372 tests
passing, `devtests/` 5 passing (~7 min). Not merged to `main`.

`superpowers:finishing-a-development-branch` has never been run on it. The
CHANGELOG's 3.2.0 section is written and dated.

Three known issues are documented rather than fixed, deliberately:
- Results depend on `dt` and do not converge (HPV prevalence 0.001/0.044/0.200
  at `dt` 1.0/0.5/0.25). **Calibrations are therefore dt-specific** — Zambia's
  is at `dt=0.25` and only valid there.
- `sim.results.n_alive` ignores per-agent multiscale weight, over-reporting
  ~3.4× at `ms_agent_ratio=100` and compounding over run length. Population
  must be computed as `people.scale[alive].sum() * pop_scale`. This is a
  reporting bug only; the dynamics are right.
- Two interventions cannot share a product (`Module vx already added`), which
  blocks routine-plus-catch-up vaccination.

Also open upstream: v3 has no equivalent of v2's HIV-mortality subtraction, so
it double-counts HIV deaths — ~9% of adult all-cause mortality in Zambia around
2000. Tracked at starsimhub/stisim#574; Robyn's call was not to fix it for 3.2.

## 5. PR for John

`v3-port` → `main` on `github.com:AndohJ0/HPVSim_Zambia` (John's repo, not an
IDM one). Currently 15 commits ahead of `main` at `c323bff`, pushed. Do (1)–(4) first: the
PR should land against released dependencies with figures that reproduce the
paper.

The PR description is the real deliverable for John, since he did all this work
in 2.2.6 and needs to understand what changed. It must say, plainly:

- **His parameter values are not portable, and why** — the ~3.2× multiscale
  bias in 2.2.6 at `ms_agent_ratio=100`, which his calibration absorbed. His
  published numbers were produced under it.
- **What that does and does not invalidate.** The fit to data is reproducible;
  the fitted parameter values are not. Supplementary Table 1 and Figure S1
  change.
- **What v3 renamed**, with the mapping: `sev_dist` → `cross_immunity.rel_sev`,
  `debut` → `debut_f`/`debut_m`, `layer_probs` → `layer_probs_marital`/`_casual`,
  the `poisson1` offset now applied by the network, CD4 strata `lt200`/`gt200`
  → `_lo`/`_hi`, `art_failure_prob` → stisim's `p_effective_art`.
- **That his two HIV mortality CSVs are now unused**, and that this is not a
  silent loss: v2 subtracted them from UN all-cause mortality to avoid
  double-counting its own Weibull HIV deaths. v3 does not, which is the
  stisim#574 issue.
- **That Figure 2's palette changed** — blue/green/red fails a colourblind
  check at deutan ΔE 5.3 between the No-ART and Status-Quo lines, whose
  divergence is the ART result. Now Okabe-Ito at ΔE 11.4.

## Environment

- `hpvsim` conda env: hpvsim 3.2.0 and stisim 1.6.1, both editable from
  `/Users/robynstuart/gf/`. Use absolute interpreter paths;
  `conda run -n` resolves to the active env on this machine.
- `starsim` conda env: hpvsim **2.2.6** from site-packages, for the v2 baseline.
  Do not install 2.2.6 anywhere that shadows the v3 editable checkout.
- Heavy runs go on `zebra` (IDM Azure, non-spot, ~160 cores). Take the core
  count from `nproc`, not from a document; check `who` first; run in `tmux`.
  `run_calibration.py` hardcodes `n_workers=80`, which idles half the machine —
  v3's Optuna JournalStorage has no sqlite deadlock, so it can go to `nproc`.

## Do not

- Do not trust anything in `figures/` or `results/zambia_figure_*.csv` until
  regenerated at full scale.
- Do not read `sim.results.hiv.n_*` age strata — hpvsim suppresses them as of
  3.2 precisely because they are raw agent counts. Use `hpv.by_age` or
  `analyzers.CancerByAgeHIV`.
- Do not read any annualised result in the sim's final year; run one year past.
- Do not call `v2_calib_pars_to_v3()` once and reuse it across sims.
- Do not commit simulation outputs; `rsync` them off the VM.
- Do not edit `data/` — including the two HIV mortality CSVs, which look unused
  but are needed for the 2.2.6 comparison.
