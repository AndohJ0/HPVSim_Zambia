# Investigation: Zambia v3 port reproduces the calibration targets but not the scenario narrative

Written 2026-09-03, after the 5000-trial recalibration in `results/zambia_calib.obj`.
Picks up where `HANDOFF.md` §1 left off: the calibration converges cleanly, but
the counterfactual scenarios do not tell the manuscript's story. Read
`HANDOFF.md` first for the environment, why parameter values are *expected* to
change (v2 → v3 multiscale bias etc.), and the acceptance table.

## Current state

- Branch: `v3-port`, pushed. `results/zambia_calib.obj` = 5120-trial, top-200
  shrunk, best mismatch **2.46**.
- Fig-gen pipeline works end-to-end:
  `python run_counterfactuals.py --run-sim --n-pars 10 --n-seeds 10 --stop 2026`
  then `python plot_figures.py`.
- `run_multi_sim` in `run_functions.py` is now parallelized across
  `(par_set, seed)` tasks; a 10×10 run finishes on the order of minutes.

## What matches, what doesn't

Fig 1 (calibration) is roughly right in shape but hides a real problem in
panel (c). Panel (a) cases-by-age tracks the GLOBOCAN targets across all bins.
Panel (b) ASR 2020 lands at ~73 per 100k against a GLOBOCAN target of 65.5 —
about 12 % high, consistent with a mismatch of 2.46. Panel (c) — cancer IRR by
age — misses badly on the age gradient: under-shoots the young end (25–35
median ~2–3 against targets 6–9) and over-shoots the older end (55–70 median
~5–6 against targets 3). The by-age rate-ratio target is not in the
calibration (`hpv.Calibration(data=)` cannot express an HIV-stratified by-age
target; see the comment in `run_calibration.py`), so the fit is uninformed
about the age gradient. Fixing the age-gradient miss will need either a
hand-rolled objective on this metric or a structural change to the model.

**Fig 2 (crude cancer incidence over time)** shows two things: a scenario
ordering that may or may not be real, and levels that need verifying.

- *Scenario ordering.* The manuscript has *Status Quo* (with ART) *below*
  *No ART* post-2010; ours has Status Quo pulling above No ART post-2015. This
  could be a genuine finding — under wider ART coverage, more WWH survive with
  suppressed HIV, and the surviving pool accumulates cervical cancer over
  decades, so Status Quo can plausibly overtake No ART in a mechanistic
  simulation. Before treating it as a bug, the mechanism needs to be verified:
  is the WWH survival curve under Status Quo consistent with real Zambia (or
  with the v2 baseline)? And if the ordering is real, the paper's narrative
  needs updating to say so carefully — "ART extends WWH life, and beyond a
  time horizon the surviving-and-treated pool carries more cancer than an
  untreated dying pool" is a very different claim from "ART reduces cancer."
- *Levels.* Manuscript Status Quo crude ~22 per 100k in 2025 vs our 37 — we
  are ~70 % too high. Sanity check: manuscript's crude ~22 with ASR ~65 gives
  an ASR/crude ratio of ~3.0, which is what you would expect for Zambia given
  the median age of ~17 and how much WHO2000 upweights older ages. Ours has
  crude 35 and ASR 73 → ratio 2.09. Our ASR is only 12 % high but our crude is
  60 % high, so the excess sits at ages where Zambia's population is dense
  (younger/middle) rather than at the ages WHO2000 upweights (60+). Fig 3(b)
  agrees: we match at 45–50 but over-shoot at 55–70.

**Fig 3(a) and 3(b)** show the same pattern of endless accumulation. Women with
HIV (WWH) cancer incidence rises through 2025 rather than peaking around 2010.
The 45–50 age bin *does* match the manuscript almost exactly (~300/100k vs
manuscript ~293), but the incidence rate keeps climbing to 1220/100k in the
80–85 bin. Numbers behind Fig 3(b), Status Quo 2025:

| Age bin | WWH cancers | WWH incidence /100k | No-HIV incidence /100k | IRR |
|---|---|---|---|---|
| 45–50 | 262 | **300** (matches) | 79 | 4.2 |
| 55–60 | 277 | 684 | 157 | 4.3 |
| 65–70 | 126 | 856 | 216 | 4.3 |
| 80–85 | **8.5** | **1220** | 140 | 8.7 |

The 80–85 rate is small-numbers driven (8.5 cancers over roughly 700 WWH), but
65–70 is real signal on a denominator of ~14,700 WWH, and it's 4× the manuscript.
So it is *not just* a denominator artefact — the WWH pool at 60+ is genuinely
carrying too much cancer.

## Hypotheses, ordered by suspicion

### H1 — Latency parameters converged on extreme values

Best pars: `hpv_control_prob = 1.0` (maximum: *every* clearance goes to a
latent state), `hpv_reactivation = 0.025`. The posterior densities on all 200
top sets are piled at those bounds. `HANDOFF.md` flagged this exactly:
`hpv_control_prob` and `hpv_reactivation` are unidentified by 2020 cancer
targets, the paper never calibrated latency, and both parameters could be
closed.

Implication: latency reactivation drives an ever-growing pool of cancer that
is largely decoupled from HIV dynamics. The post-2015 rise in Status Quo may
be latency reactivation, not the HIV+ART interaction.

**Test.** Drop `hpv_control_prob` and `hpv_reactivation` from `calib_pars` in
`run_calibration.py`. Leave the make_sim defaults (`hpv_control_prob=0`,
`hpv_reactivation=0.025`), which make latency a no-op. Recalibrate and rerun
counterfactuals.

**Success criterion.** Fig 2 Status Quo trajectory bends downward relative to
No ART post-2010. Fig 3(b) WWH incidence peaks around 45–50 rather than
climbing indefinitely.

### H2 — Truncated calibration bounds

The posterior densities show roughly ten calibration parameters piled at their
bounds: `beta` (low), `age_risk.age` (high, 44/45), `cross_immunity.own_imm_hr`
(high, 1.0), `network.m_cross_layer` (high), `hiv.rel_sus_lo`/`_hi` (low),
`hiv.rel_sev_hi` (low), `hiv.rel_reactivation_hi` (low). The fit wants to go
further than the current bounds allow.

**Test.** Widen the flagged bounds by 30–50 % on the pinched side and
recalibrate. Compare the new posterior densities: parameters that shift when
released were being truncated; parameters that stay put weren't.

**Success criterion.** Best mismatch drops meaningfully (say, below 2.0). If
mismatch drops without narrative improvement, bound truncation was not the
scenario story.

### H3 — HIV mortality / ART survival

Under Status Quo, WWH are surviving to accumulate cancer through 80+. In the
manuscript, WWH incidence is essentially zero after 65. `HANDOFF.md` flags
`stisim#574`: v3 lacks v2's HIV-mortality subtraction, so v3 should have *more*
HIV deaths, not fewer. So the mechanism isn't missing HIV mortality directly.

Candidates: `p_effective_art = 0.9` (Zambia's default) may be higher than what
v2 modelled effectively, keeping too many WWH virally suppressed → too many
alive at 60+. Or stisim's CD4 progression under ART differs from the v2 Weibull
mortality that the manuscript's calibration used.

**Test.** Instrument `make_sim` to run one WWH cohort under Status Quo and print
the survival curve, ART-suppression fraction, and CD4 trajectory by age. Compare
to what v2 produces via `git worktree add ../zambia-v2 35c647d~1` and the
`starsim` conda env's hpvsim 2.2.6. `HANDOFF.md` §1's "Getting the v2 code and
environment" block has the setup.

**Success criterion.** Under Status Quo, WWH survival curves match v2 within
~10 % out to age 60. If they diverge sharply, ART efficacy or CD4 progression is
the culprit.

### H4 — HIV incidence at older ages

If the imposed HIV incidence data has non-zero rates at ages 60+, we would
generate late-onset infections → older WWH → cancer at old ages. v2 may have
truncated the incidence data at, e.g., age 55.

**Test.** `head data/zambia_hiv_incidence_updated.csv` and look at incidence
rates at age ≥ 60. If they're nonzero, plot against age and compare to the
raw HIV surveillance data. If v2 used a truncated version, restore the
truncation.

**Success criterion.** Age of last non-trivial incidence in the data matches
what v2 fed the calibration.

### H5 — Multiscale bias residual

`HANDOFF.md` documents a ~3.2× multiscale bias in v2 at `ms_agent_ratio=100`
that the v2 calibration absorbed. v3 is unbiased. This changes *absolute*
levels but should not change scenario *direction*. Only investigate if H1–H4
fail.

## Investigation order

1. **H1: close latency.** Fastest test, most concrete lever. About 30 minutes to
   recalibrate and rerun figures on `zebra`.
2. **H2: widen bounds.** Only if H1 changes the shape but the fit remains bad.
   Another ~30 minutes to recalibrate.
3. **H3: HIV survival.** Deeper investigation, needs the v2 baseline as a
   comparison. Expect a day.
4. **H4: HIV incidence data.** Quick check, low probability but almost free to
   rule out.

## Reproduce the current state

```bash
# Zebra, HPVSim_Zambia repo, v3-port branch
python run_calibration.py                         # 5000 trials, ~30 min
python run_counterfactuals.py --run-sim --n-pars 10 --n-seeds 10 --stop 2026
python plot_figures.py
```

Compare `figures/*.png` to `docs/figures_original/*.png`.

## Tools and constraints

- Zebra is idle at the time of writing; `nproc = 160`. Run heavy jobs in `tmux`
  and check `who` first.
- Calibration writes to `results/zambia_calib.obj` — 200-set shrunk. Sims are
  regenerable via `hpv.make_calib_sims`.
- Counterfactuals write `results/zambia_figure_{timeseries,by_age}.csv`. Both
  are gitignored.
- Do not touch `data/*_hiv_mortality*.csv` — needed for the v2 baseline.
- Do not read annualised results in the sim's final year; `--stop 2026` is
  deliberate so 2025 is complete.
- Do not commit simulation outputs; `rsync` them off the VM.

## Success criteria for the whole port

Per `HANDOFF.md` §1's acceptance table, and specifically for the narrative:

- Status Quo *below* No ART post-2010 in Fig 2, both peaked around 2010,
  declining to 2025.
- WWH cancer incidence peaks at 45–50 (~293/100k) in Fig 3(b), women without
  HIV peak at 60–65 (~50/100k), WWH near zero at 65+.
- Cumulative cases by 2025 Status Quo ≈ 71,042; HIV attributable fraction
  ≈ 49.9 %.

## Deliverable

When the investigation converges (or is stopped), write a short summary in this
file describing what was tried, what worked, and what the new port state looks
like. Then move to `HANDOFF.md` §2 (README, PR readiness) and §5 (the PR).
