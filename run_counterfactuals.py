"""Run the paper's three counterfactual scenarios and save figure data.

The heavy simulation and the plotting are deliberately separate: run this on a
VM with --run-sim to produce tidy CSVs in results/, then make the figures
locally with plot_figures.py, which needs nothing but those CSVs.

Scenarios, per the Methods:
  status_quo  -- HIV and ART as observed; no vaccination, screening or treatment
  no_hiv      -- HIV never introduced (Scenario 1)
  no_art      -- HIV as observed, ART coverage set to zero (Scenario 2)

Usage:
    python run_counterfactuals.py --run-sim                  # on the VM
    python run_counterfactuals.py --run-sim --n-pars 50 --n-seeds 100
"""

import argparse

import numpy as np
import pandas as pd
import sciris as sc

from analyzers import CancerByAgeHIV
from run_functions import get_top_calibrated_pars, run_multi_sim

LOCATION = 'zambia'

# Scenario -> make_sim kwargs. Scenario 1 removes HIV entirely rather than
# zeroing incidence, so CD4 dynamics and ART go with it.
SCENARIOS = {
    'status_quo': dict(model_hiv=True, art_coverage_scale=1.0),
    'no_hiv': dict(model_hiv=False, art_coverage_scale=1.0),
    'no_art': dict(model_hiv=True, art_coverage_scale=0.0),
}

# All-age annual time series (Figures 2 and 3a). These now come straight from
# hpvsim: as of v3.2 the HIV-stratified rates use a female denominator and are
# annualised per calendar year, so no local recomputation is needed.
TIMESERIES_METRICS = [
    'cancer_incidence',            # crude, per 100k women -- Figure 2
    'cancer_incidence_with_hiv',   # Figure 3a
    'cancer_incidence_no_hiv',     # Figure 3a
    'asr_cancer_incidence',        # age-standardised, for reference
    'new_cancers',                 # for cumulative-case totals
    'cancer_rate_ratio',
]

# Years reported by the age-stratified analyzer: 2020 for the calibration
# panels (Figure 1), 2025 for the age-specific comparison (Figure 3b).
REPORT_YEARS = (2020, 2025)

# Figure 1a compares against GLOBOCAN, whose bins start with a wide 0-15.
GLOBOCAN_EDGES = np.array([0., 15., 20., 25., 30., 35., 40., 45., 50.,
                           55., 60., 65., 70., 75., 80., 85., 100.])


def _timeseries_rows(sims, scenario):
    """One row per (scenario, parameter set, year, metric), seeds averaged.

    Seeds are collapsed within a parameter set so the spread that survives to
    the figures is parameter uncertainty, which is what the published
    interquartile bands represent. Keeping every seed would multiply the file
    by n_seeds for no gain in the plots.
    """
    frames = []
    for sim in sims:
        res = sim.results.all_hpv
        years = np.floor(sim.results.timevec.years).astype(int)
        row = {'scenario': scenario, 'par_set': getattr(sim, 'rank', np.nan),
               'year': years}
        data = {m: np.asarray(res[m], dtype=float) for m in TIMESERIES_METRICS
                if m in res}
        frames.append(pd.DataFrame({**row, **data}))
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True).melt(
        id_vars=['scenario', 'par_set', 'year'], var_name='metric', value_name='value')
    # A calendar year holds several timesteps carrying the same annual value,
    # and several seeds; mean over both collapses to one value per par set.
    return (long.groupby(['scenario', 'par_set', 'year', 'metric'], as_index=False)['value']
            .mean())


def _by_age_rows(sims, scenario):
    """One row per (scenario, parameter set, year, age bin, metric)."""
    frames = []
    for sim in sims:
        analyzer = next((a for a in sim.analyzers.values()
                         if isinstance(a, CancerByAgeHIV)), None)
        if analyzer is None:
            continue
        for year in REPORT_YEARS:
            try:
                df = analyzer.to_dataframe(year)
            except ValueError:
                continue  # year outside this sim's window
            df = df.assign(scenario=scenario, par_set=getattr(sim, 'rank', np.nan),
                           year=year)
            frames.append(df)
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True).melt(
        id_vars=['scenario', 'par_set', 'year', 'bins'],
        var_name='metric', value_name='value')
    return (long.groupby(['scenario', 'par_set', 'year', 'bins', 'metric'],
                         as_index=False)['value'].mean())


def run(n_pars=50, n_seeds=100, stop=2026, debug=0, batch_size=25):
    """Run every scenario and write results/zambia_figure_{timeseries,by_age}.csv.

    stop defaults to 2026 so that 2025 -- the last year the figures report --
    is a complete calendar year. Annualised results read low in a partly
    covered final year, since a fraction of the year's events is divided by a
    full year of person-time.
    """
    calib = sc.loadobj(f'results/{LOCATION}_calib.obj')
    top_pars = get_top_calibrated_pars(calib, n=n_pars)
    if len(top_pars) < n_pars:
        print(f'NOTE: asked for {n_pars} parameter sets, calibration holds '
              f'{len(top_pars)}; using all of them.')

    ts, by_age = [], []
    for name, kwargs in SCENARIOS.items():
        print(f'\n=== scenario: {name} ({len(top_pars)} par sets x {n_seeds} seeds) ===')
        sims, _ = run_multi_sim(
            top_pars=top_pars, end=stop, n_runs=n_seeds, batch_size=batch_size,
            debug=debug, create_reduced=False, verbose=0.0,
            analyzers=[CancerByAgeHIV(years=REPORT_YEARS, edges=GLOBOCAN_EDGES)],
            **kwargs,
        )
        for collect, fn in ((ts, _timeseries_rows), (by_age, _by_age_rows)):
            rows = fn(sims, name)
            if rows is not None:
                collect.append(rows)
        del sims

    for frames, stem in ((ts, 'timeseries'), (by_age, 'by_age')):
        if not frames:
            continue
        out = f'results/{LOCATION}_figure_{stem}.csv'
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f'wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-sim', action='store_true',
                        help='run the scenarios (heavy; intended for a VM)')
    parser.add_argument('--n-pars', type=int, default=50,
                        help='calibrated parameter sets to use (default 50)')
    parser.add_argument('--n-seeds', type=int, default=100,
                        help='seeds per parameter set (default 100)')
    parser.add_argument('--stop', type=int, default=2026,
                        help='final sim year; must exceed the last reported year')
    parser.add_argument('--debug', type=int, default=0,
                        help='1 for a small, fast, scientifically meaningless run')
    args = parser.parse_args()

    if not args.run_sim:
        parser.error('nothing to do: pass --run-sim to run the scenarios. '
                     'Figures are made separately with plot_figures.py.')

    T = sc.timer()
    run(n_pars=args.n_pars, n_seeds=args.n_seeds, stop=args.stop, debug=args.debug)
    T.toc('Done')
