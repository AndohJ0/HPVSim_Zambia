# Zambia cervical screening study on HPVsim v3

## Purpose and current state

Develop a lean, reproducible implementation of the Zambia cervical screening
strategies. Use the existing Zambia simulation factory and supported HPVsim v3
interfaces. Production simulations run on HPC; local runs validate mechanics.

This file records agreed scope and an ordered execution plan. It does not imply
that unresolved scientific assumptions have been approved or that runs have
already been performed. Read this file before every task.

The intended development branch is `cea-v3-development`. When this file was
created, the checkout was `cea-development`, with no local branch named
`cea-v3-development`. This documentation-only task leaves the branch unchanged.
At the first implementation task, inspect Git status and branches, then use the
intended branch. If absent, create it from the reviewed current checkout without
resetting, discarding changes, or modifying `main`. Report the starting commit.

## Non-negotiable boundaries

- Do not edit, replace, symlink, or monkey-patch installed HPVsim, Starsim,
  STIsim, or other dependency packages. Use public interfaces, custom product
  dataframes, eligibility callbacks, and small project-local analyzers or
  subclasses where required. Explain any local extension and test it.
- Do not copy the legacy environment, duplicate the disease engine, or import
  code from the legacy project at runtime. The v3 project must run independently.
- Exclude vaccination from every strategy and both comparators. Do not add it
  as a sensitivity analysis unless the user changes this scope.
- Keep the shared calibrated natural-history model consistent across arms.
  Do not silently alter HIV progression, mortality, latency, network parameters,
  multiscale settings, or calibration targets to make scenarios behave as expected.
- Do not assume translated v2 parameters reproduce a valid v3 calibration.
- Preserve source inputs and calibration artifacts. Never overwrite them with
  new runs. Keep generated outputs separate from source data.
- Do not launch production grids locally. Do not infer HPC access, account,
  resource limits, or approval for a production submission from this file.
- Never fabricate results, precision, citations, run completion, or test success.

## Keep the repository lean

Reuse `run_functions.py`, `analyzers.py`, `run_calibration.py`, and existing tests.
Aim for two new study scripts:

1. `run_cea.py`: strategy definitions, product loading, simulation construction,
   local smoke runs, and execution of individual HPC tasks.
2. `analyze_cea.py`: run validation, paired scenario comparisons, aggregation,
   tables, and figures. Economic post-processing may be added here when specified.

Use `data/` for the small set of versioned study inputs, `tests/` for meaningful
tests, `hpc/` for scheduler wrappers, and `docs/` for study documentation. The
two-script goal does not prohibit tests, an HPC wrapper, or a justified analyzer.
Avoid one script per strategy and avoid broad refactors of existing code.

## Agreed arms

| ID | Primary test | Triage | Screening schedule |
|---|---|---|---|
| `comparator` | None | None | No cervical screening |
| `current_practice` | VIA, provisional representation | None | Baseline uptake and schedule require resolution |
| `s1a` | VIA | None | Ages 30 to under 65, every 5 years, both HIV groups |
| `s1b` | VIA | None | HIV-negative: 30 to under 65, every 5 years; WLHIV: 25 to under 65, every 3 years |
| `s1c` | VIA | Restricted-panel HPV | Same differentiated schedule as s1b |
| `s2a` | HPV DNA | None | Same universal schedule as s1a |
| `s2b` | HPV DNA | None | Same differentiated schedule as s1b |
| `s2c` | HPV DNA | LBC ASC-US+ | Same differentiated schedule as s1b |

The upper age bound above makes the legacy `[30, 65]` / `[25, 65]` convention
explicit; confirm this interpretation when locking schedules. S2d, S2e, S2f,
CD4-based screening strategies, and clinical cancer-stage extensions are outside
the initial scope. Universal schedules still use HIV-specific test and treatment
products. HIV status changes during simulation; screening history must survive
movement between streams.

Retain both comparators. The proposed primary policy reference is current
practice, with no screening as a secondary counterfactual. Confirm this reporting
hierarchy when finalizing baseline assumptions. No screening does not mean no ART
or background medical care.

## Resolve scientific assumptions before production

Track decisions in one concise `docs/cea_decisions.md`, created during Stage 1.
For each item record the question, source/denominator, recommendation, user
decision, and affected inputs/tests. Keep unresolved items explicitly open.
Existing legacy code and notes are evidence of prior implementation, not automatic
scientific approval for v3. Do not ask again about decisions already made here or
subsequently by the user.

1. Baseline uptake: the cited Lubeya et al. 2024 ZAMPHIA analysis reports 22.2%
   ever screened among women aged 15–49, not an annual probability or a verified
   27% target. Verify survey weighting and age/HIV denominators before fitting.
   Use time since last screen if available. Lifetime uptake does not uniquely
   identify annual uptake; any conversion requires explicit exposure assumptions.
2. Scale-up: distinguish annual attendance while due, participation per screening
   round, recent/up-to-date coverage, and WHO screening by ages 35 and 45. Do not
   pass 0.70 as an annual probability merely because the policy target is 70%.
   Agree the target measure and fit/verify the input that achieves it by 2030.
3. History: agree common pre-2025 screening history or an explicitly simplified
   common starting state. Never give only the comparator a screening head start.
   Review whether VIA-only is a suitable current-practice approximation.
4. Diagnostic endpoints: distinguish detection of HPV infection from detection
   of CIN2+. Do not assign one minus clinical CIN2+ specificity directly to
   HPV-uninfected women without justification. Map study endpoints to model states.
5. Products: preserve separate HIV-specific diagnostic and treatment inputs,
   subject to evidence review. Verify primary versus conditional triage accuracy,
   restricted-panel genotype approximations, and person-level results in mixed
   genotype/state cases. `genotype='all'` alone does not ensure one draw per woman.
6. Visits: distinguish annual screening uptake from per-visit completion. Check
   the pinned runtime's probability convention; routine triage may require
   `annual_prob=False`. Preserve the intended same-visit and referral structure;
   do not confuse a probability gate with elapsed delay. Document reflex LBC's
   sample assumption and post-treatment follow-up scope.
7. HIV/ART: decide post-data projection assumptions using the v3 data contract.
   Do not transplant v2's imposed mortality input into v3's endogenous HIV mortality.
8. Horizon: legacy evaluation is 2025–2050 with quarterly timesteps(v3 is not quarterly). Confirm the
   complete reporting window, intervention stop, follow-up horizon, and whether
   a longer-horizon sensitivity is needed. Do not truncate the final year silently.
9. Calibration and uncertainty: assess the v3 fit before production, including
   informative HIV cancer rate ratios. Agree ranks, replicate seeds, agent count,
   and multiscale settings from validation and HPC benchmarks. The old 100-by-50
   grid is a reference, not an approved resource commitment. Top-ranked Optuna
   fits are not automatically Bayesian posterior samples. Reconcile the legacy
   decision log's `cancers 2 : ASR 1 : IRR 1` description with the implemented
   `cancers 1.5 : ASR 0.5 : IRR 1` target totals before launching a new
   production calibration; do not infer the objective used for an artifact from
   a later comment.

Costs, discounting, DALYs, thresholds, cancer-management access, cost lag and stage
costs require a separate economic specification if requested. Do not delay the
agreed health-outcome pipeline to build an unrequested economic extension.
Never imply that a costing overlay models cancer diagnosis, stage shift or survival.

## Calibration methodology

Reproduce the useful behavior of the improved legacy calibration through
project-local code only. Do not copy or modify the legacy HPVsim
`calibration.py` or `analysis.py`, and do not patch the installed v3 package.
Use the stock v3 `hpv.Calibration` with a local analyzer and evaluation function.

Treat the legacy improvement as a combination of corrected HIV-stratified
outputs, pooled rate calculations, target preprocessing, explicit objective
weights and a new calibration run. Do not attribute the improved fit to the
optimizer or to `calibration.py` alone. Keep any v2 HIV/model corrections
separate from the v3 calibration design and verify whether v3 already implements
the intended behavior before proposing a local extension.

For the primary calibration objective:

- Fit age-specific cervical cancer cases over ages 25 to 75 inclusive, subject
  to confirmation against the target bins.
- Fit the age-standardized cervical cancer incidence rate as a correlated
  summary of the same cancer burden, rather than treating it as fully independent
  information.
- Fit HIV-positive versus HIV-negative cervical cancer incidence rate ratios
  over ages 25 to 60 inclusive. Pool five calendar years ending in the target
  year unless source review supports a different observation window.
- Calculate each pooled stratum-specific incidence rate from summed incident
  cancer events divided by summed female person-time. Take the ratio only after
  pooling numerators and denominators; never average annual rate ratios.
- Use simulation weights and population scaling exactly once. Person-time for a
  timestep is the eligible female population multiplied by the timestep length.
- Remove missing observed values before normalizing or weighting the objective.
  Match every retained observation explicitly by target name, year and age bin;
  never depend on CSV row order.
- Assign a declared total weight to each target family and divide that total
  across its usable observations. The provisional preferred structure is cancer
  cases 1.5, ASR 0.5 and HIV rate ratios 1.0, so the combined cancer-burden
  family has twice the weight of the HIV gradient. Confirm this after resolving
  the legacy record discrepancy and examine reasonable alternatives as a
  sensitivity analysis.
- Calculate the HIV rate-ratio component on the log scale, for example the mean
  or weighted mean of
  `abs(log(IRR_model) - log(IRR_observed))`. This makes reciprocal proportional
  errors comparable. Require strictly positive finite ratios. If source counts,
  person-time or uncertainty intervals become available, prefer an appropriate
  likelihood or inverse-variance formulation and document the replacement.
- Never silently discard a required bin because its modeled denominator is zero
  or its ratio is non-finite. Apply a declared finite penalty or reject the trial,
  and report the frequency and affected bins. Test the penalty's influence.
- Save the total objective and separate components for cancer cases, ASR, HIV
  rate ratios and missing-bin penalties for every retained trial. Raw mismatch
  values are comparable only when target preprocessing, weights and loss
  definitions are identical.

Use the following notation in the implementation and documentation. For HIV
stratum `h`, age bin `a` and pooled timestep set `T`, calculate
`rate[h,a] = 100000 * sum_t(C[h,a,t]) / sum_t(N[h,a,t] * dt)`, followed by
`IRR[a] = rate[HIV+,a] / rate[HIV-,a]`. For valid retained IRR bins, calculate
`L_IRR = sum_a(w[a] * abs(log(IRR_model[a]) - log(IRR_observed[a])))`, where
the bin weights sum to the declared IRR-family weight. The complete objective is
`L_total = L_cases + L_ASR + L_IRR + L_missing`. Record the exact cancer-case
and ASR loss functions supplied by the selected v3 runtime; do not describe them
as log-scale losses unless they are explicitly changed and validated.

Use a two-stage stochastic calibration workflow when computationally feasible.
Run the broad Optuna search with a controlled common seed and then rerun the
leading parameter sets across several prespecified seeds. Rank or select the
calibrated ensemble using both target-specific fit and across-seed stability.
Do not describe the retained ranks as posterior samples. Compare the new and
existing calibrations using target-level tables and plots on matched seeds and
settings, rather than comparing total objective values from different loss scales.

Calibration tests must cover target filtering and family-weight totals, exact
year/age alignment independent of row order, pooled event and person-time
calculations, log-scale IRR loss, zero-denominator and non-finite penalties,
HIV-stratified cancer reconciliation with total cancers, fixed-seed repeatability,
and a small smoke run through the installed stock v3 calibration interface.

## Ordered execution stages

### Stage 1 — Lock the study specification

Read the current project and relevant legacy sources. Resolve coverage, baseline,
product endpoint and visit assumptions in small discussions. Verify original
references rather than copying citations from previous generated documents.
Produce a compact decision record and source-to-parameter table. Do not implement
unresolved scientific choices as final defaults.

### Stage 2 — Verify the v3 runtime and establish baseline tests

Record exact Python, HPVsim, Starsim, STIsim and other relevant dependency
versions and import paths. Inspect the corresponding source/API, not merely
upstream `main`; record the commit when using a source installation. Prepare a
reproducible environment specification. Never use the old v3.0 environment as
proof that this repository's >=3.2 requirements work.

Run the existing tests and report actual outcomes. Establish a short baseline
run with the intended timestep, because annual debug runs can conceal quarterly
probability or timing problems. Do not modify calibration just to pass smoke tests.

Then implement and test the project-local calibration analyzer, target
preprocessing and evaluation function described above. Run a small diagnostic
calibration before any full HPC search. Inspect component losses, missing-bin
frequency and parameter identifiability; a lower total mismatch alone is not an
acceptance criterion. Preserve existing calibration artifacts and write new runs
to versioned paths with their objective specification, input hashes and runtime
versions.

### Stage 3 — Implement and test products and cascades

Implement fresh per-simulation products and the agreed arms through supported
interfaces. Add tests for the following scientific contracts:

- Allowed arms, no vaccination, ages, intervals and zero comparator screening.
- No overlap between HIV streams at an event; history persists across seroconversion.
- Valid diagnostic probability groups, explicit result hierarchy, treatment
  efficacy bounds, and expected person-level performance by HIV status.
- No genotype/state compounding inconsistent with the selected test definition.
- Triage is restricted to current eligible upstream positives; assignment and
  treatment follow their intended branches, with no stale outcomes or duplicate care.
- Correct per-visit completion at quarterly timesteps; same-day versus delayed
  treatment is represented as specified. Avoid trivial tests at probability 1 only.
- Service counts include delivered ineffective treatment and relevant false-positive
  pathways. Counts respect person weights and population scaling exactly once.
- Analyzers do not change model trajectories. HIV-stratified incident cancers
  reconcile with total incident cancers under the declared event definition.
- Fixed-seed repeatability, independent run state, paired scenario identifiers,
  and zero treatment efficacy as a mechanistic negative control where appropriate.

Use deterministic synthetic cases for mechanics and appropriately sized stochastic
checks for probabilities. Do not assert that every individual stochastic screening
run must have fewer cancers than its comparator. Never clip negative cases averted.

### Stage 4 — Prepare and pilot HPC execution

The existing `hpc/` scripts contain legacy paths and are not ready-to-use templates.
Obtain the actual cluster work directory, environment setup, scheduler/account,
partition, resource limits and transfer method before finalizing submission.

Create a deterministic manifest keyed by arm, calibration rank and replicate.
Use matching rank/seed pairs across arms. Record source commit, dependency versions,
calibration/input hashes, full study settings and completion status per task.
Avoid mutable shared RNG objects and oversubscribed nested parallelism.

Use unique task logs and outputs, atomic completion writes, and resumable execution.
Skip only outputs whose completion and metadata validate; do not mix incompatible
runs. Benchmark a small representative HPC pilot, including a triage arm. Estimate
runtime, memory and storage before presenting a concrete production submission.
Prepare the scripts and manifest before requesting any missing submission approval.

### Stage 5 — Run and validate production

Submit the agreed grid only when HPC access and submission scope are authorized.
Monitor according to the user's instructions; record job IDs and failures. Retry
failed or incomplete cells without overwriting valid outputs. Validate the full
manifest before scientific aggregation. Label partial analyses explicitly.

### Stage 6 — Health outcomes and paired comparisons

Export tidy annual and cumulative results for every arm, overall and separately
for women with HIV and women without HIV. Primary required outcomes are incident
cervical cancers and cases averted versus each comparator. Include incidence rates
with explicit female denominators, and cancer deaths if validly supported by the
runtime and analyzer. Export screening, triage and treatment volumes as validation
and future costing inputs.

Default proposed HIV grouping is status at cancer onset (or at death for death
outcomes), not status at simulation end. State and confirm the rule. These are
population-group differences, not necessarily direct causal effects confined to
the same women or to their initial HIV group.

Compute cases averted as comparator minus strategy within matching calibration
rank and seed before summarizing uncertainty. Report absolute and percentage
effects; return undefined for a percentage with zero comparator denominator.
Do not subtract independently calculated quantiles. Specify the uncertainty
summary and distinguish stochastic from calibration uncertainty. Preserve raw
counts as well as any subsequently approved discounted outcomes.

### Stage 7 — Final scientific documentation

Maintain methods descriptions as implementation is validated. Populate results
only after complete production-output validation. Deliver:

1. A test suite and concise validation report with commands, runtime versions,
   pass/fail outcomes, and unresolved limitations.
2. Scenario-level health outcome tables/figures, including cancers and cases
   averted for both HIV populations and overall, against both comparators.
3. An updated technical appendix covering natural history, HIV/ART, calibration,
   coverage definitions, product mappings, cascade details, equations, inputs,
   HPC execution, uncertainty, validation and limitations.
4. A separate refined Methods and Results appendix suitable for a scientific
   paper. Methods must describe the implemented study; Results must use verified
   outputs, with matched denominators, horizons and uncertainty labels.

Use concise Markdown working drafts in `docs/`; produce the two final appendices
as Word deliverables using the documents skill with rendering and visual
verification at the documentation stage. Keep results placeholders visibly unfilled until runs finish.
Verify references; do not call implementation assumptions measured evidence.

Both appendices must state the calibration observation windows, retained age
ranges, target-family weights, missing-data rules, bin-alignment rules, stochastic
seed procedure and ensemble-selection rule. Include the pooled incidence and
rate-ratio equations and the exact log-scale loss equation. Explain that the
log transform compares proportional departures, identify any finite penalty for
undefined modeled ratios, and report sensitivity to that penalty and to plausible
target-family weights. The technical appendix must additionally provide the
component-loss diagnostics and reproducibility metadata. The paper Methods must
describe the final prespecified procedure concisely; its Results must report the
achieved fit separately for cancer cases, ASR and the HIV gradient, without
interpreting Optuna ranks as posterior uncertainty.

## How to execute each task

Work on the stage named in the user's prompt. Inspect existing work before editing;
do not restart completed stages. Complete routine authorized implementation and
verification without repeated permission requests. Ask concise questions only for
missing scientific choices or external execution details that affect the result.
Continue independent work while awaiting those answers.

At the end of a task report: what changed, what was tested, evidence of success
or failure, remaining scientific decisions, and the next concrete stage. A test
not run is not a passed test. HPC jobs submitted are not completed results.

## Source locations

- Current project: this repository, especially `run_functions.py`, `analyzers.py`,
  `run_calibration.py`, `run_top_calibrations.py`, `data/`, and `tests/`.
- Legacy reference only: `../hpvsim_zam/cost_effectiveness/cea/` and
  `../hpvsim_zam/cost_effectiveness/docs/`, including the methodology, diagnostic
  appendix, treatment tables, cost tables and `DECISION_LOG.md`. Paths in legacy
  documents may be obsolete. Do not modify the legacy project.
- Upstream cascade example:
  https://github.com/starsimhub/hpvsim/blob/main/tests/test_interventions_cascade.py
- Upstream product and delivery code: HPVsim `hpvsim/products.py`,
  `hpvsim/interventions.py`, `hpvsim/data/products_dx.csv`; Starsim
  `starsim/interventions.py`. Match inspection to the selected installed version.
- Screening uptake: Lubeya et al. 2024,
  https://doi.org/10.1177/10732748241307361
- WHO screening milestone:
  https://www.who.int/initiatives/cervical-cancer-elimination-initiative
