"""
Calibrate the Zambia model to HIV and HPV outcomes
"""

# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv
import pylab as pl

# Imports from this repository
import run_functions as rf


# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    'run_calibration',  # Make sure this is uncommented if you want to _run_ the calibrations (usually on VMs)
    # 'plot_calibration',  # Make sure this is uncommented if you want to _plot_ the calibrations (usually locally)
]
debug = False  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

# Run settings for calibration (dependent on debug)
n_trials = [1000, 10][debug]  # How many trials to run for calibration
n_workers = [80, 1][debug]  # How many cores to use


########################################################################
# Run calibration
########################################################################
def run_calib(location=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):
    """Run an Optuna-based calibration for `location` and optionally save/plot it."""
    assert location is not None, 'location must be specified'
    dflocation = location.replace(' ', '_')

    sim = rf.make_sim(calib=True, debug=debug)

    # Targets. v3 replaced datafiles= with data=; long-format CSVs
    # (year, name, age, sex, genotype, value) are parsed by hpv.data.loaders,
    # which routes a file with an age column to all_hpv.<name>.<bin> and one
    # without to the scalar all_hpv.<name>.
    #
    # data/{loc}_cancer_rate_ratios.csv is deliberately NOT included: its
    # `cancer_hiv_rate_ratios` name has no v3 equivalent that data= can reach.
    # v3 only exposes an all-age all_hpv.cancer_rate_ratio, and age-stratified
    # targets are served by an auto-created by_age analyzer whose key whitelist
    # has no HIV-stratified entries. Fitting it needs a custom eval_fn built on
    # analyzers.CancerByAgeHIV -- see hpvsim's tests/regression/calibrate_rwanda.py
    # for the hand-rolled-objective pattern.
    data = hpv.data.load_calib_data([
        f'data/{dflocation}_cancer_cases.csv',  # Globocan
        f'data/{dflocation}_asr_cancer_incidence.csv',
    ])

    # Each bound is [guess, low, high, step]. v3 requires nested-by-scope keys
    # (flat dotted keys are rejected): genotype names, 'hiv', 'network' and
    # 'cross_immunity' scope to those modules, and bare keys broadcast.
    calib_pars = dict(
        beta=[0.05, 0.02, 0.5, 0.02],
        hpv_control_prob=[0, 0, 1, 0.25],
        hpv_reactivation=[0.025, 0, 0.1, 0.025],
        age_risk=dict(risk=[3.2, 1, 4, 0.1],
                      age=[38, 30, 45, 1]),
        cross_immunity=dict(own_imm_hr=[0.5, 0.25, 1, 0.05]),
    )

    # Sexual behaviour. layer_probs and the cross-layer probs are ANNUAL in
    # v3, whereas the v2.2.6 bounds below were per-timestep, so they are
    # converted to keep the search over the same region of behaviour space.
    # No step on the converted bounds: v2's 0.05 grid does not survive the
    # non-linear conversion, and an off-grid step makes Optuna silently
    # truncate the upper bound.
    dt = [0.25, 1.0][debug]
    to_annual = lambda p: round(float(rf._to_annual_prob(p, dt)), 4)
    sexual_behavior_pars = dict(
        network=dict(
            m_cross_layer=[to_annual(0.3), to_annual(0.1), to_annual(0.7)],
            f_cross_layer=[to_annual(0.4), to_annual(0.05), to_annual(0.7)],
            # Poisson rates, not probabilities -- no conversion. v2's poisson1
            # offset is applied by the network in v3, so these are bare lambdas.
            m_partners_casual=[0.5, 0.1, 0.6, 0.05],
            f_partners_casual=[0.2, 0.1, 0.6, 0.05],
        ),
    )
    calib_pars = sc.mergedicts(calib_pars, sexual_behavior_pars)

    # v2 passed HIV parameters as a separate hiv_pars= argument; v3 folds them
    # into calib_pars under the 'hiv' scope, with the CD4 strata flattened
    # from lt200/gt200 to _lo/_hi.
    calib_pars['hiv'] = dict(
        rel_sus_lo=[2.25, 2, 5, 0.25],
        rel_sus_hi=[2.25, 2, 4, 0.25],
        rel_sev_lo=[1.5, 1.25, 5, 0.25],
        rel_sev_hi=[1.5, 1.25, 3, 0.25],
        # v3 split v2's single rel_reactivation_prob by CD4 stratum; both are
        # opened here to preserve the v2 search range.
        rel_reactivation_lo=[3, 2, 5, 0.5],
        rel_reactivation_hi=[3, 2, 5, 0.5],
    )

    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        data=data,
        total_trials=n_trials, n_workers=n_workers,
        label=f'{location}_calib',
    )
    calib.calibrate()
    filename = f'{location}_calib{filestem}'
    if do_plot:
        os.makedirs('figures', exist_ok=True)
        hpv.plot_calibration(calib)
        pl.savefig(f'figures/{filename}.png')
    if do_save:
        # shrink() drops the full sims, leaving a small object that stays
        # loadable (and committable) without the original environment.
        sc.saveobj(f'results/{filename}.obj', calib.shrink(n_results=50))

    print(f'Best pars are {calib.best_pars}')

    return sim, calib


########################################################################
# Load pre-run calibration
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, filestem=''):
    """Load a saved calibration for `location`, optionally plotting and saving best pars."""
    assert location is not None, 'location must be specified'
    fnlocation = location.replace(' ', '_')
    filename = f'{fnlocation}_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')
    if do_plot:
        sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
        sc.options(font='Libertinus Sans')
        # v3: Calibration.plot() is the inherited starsim one, which needs
        # components/check_fit and returns nothing useful on the data= path.
        fig = hpv.plot_calibration(calib)
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        os.makedirs('figures', exist_ok=True)
        fig.savefig(f'figures/{filename}.png')

    if save_pars:
        # v3 removed trial_pars_to_sim_pars; the par set for a trial is read
        # off calib.df directly.
        top = rf.get_top_calibrated_pars(calib, n=which_pars + 1)
        sc.save(f'results/{location}_pars{filestem}.obj', top[which_pars]['pars'])

    return calib


def plot_extra_results(calib, start_year=1985, year=2020):
    """Plot cancers by HIV status over time, cancer incidence by age, and ASR incidence.

    v2 read these off calib.extra_sim_results, populated via the
    extra_sim_result_keys= argument. v3 has neither, so the best-fit
    parameters are re-run with the age/HIV analyzer attached.
    """
    best = rf.get_top_calibrated_pars(calib, n=1)[0]['pars']
    analyzer = rf.create_age_analyzer(year=year)
    sim = rf.make_sim(calib_pars=best, stop=year, analyzers=[analyzer], debug=debug)
    sim.run()

    res = sim.results.all_hpv
    years = sim.results.timevec.years
    keep = years >= start_year
    by_age = [a for a in sim.analyzers.values()
              if isinstance(a, rf.CancerByAgeHIV)][0].to_dataframe(year)

    fig, axes = pl.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(years[keep], res['cancers_with_hiv'][keep], label='HIV+')
    axes[0].plot(years[keep], res['cancers_no_hiv'][keep], label='HIV-')
    axes[0].plot(years[keep], res['new_cancers'][keep], label='Total')
    axes[0].set_title('Cancers over time')
    axes[0].legend()

    axes[1].plot(by_age['bins'], by_age['cancer_incidence_with_hiv'], label='HIV+')
    axes[1].plot(by_age['bins'], by_age['cancer_incidence_no_hiv'], label='HIV-')
    axes[1].set_title(f'Cancer incidence by age, {year}')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].legend()

    axes[2].plot(years[keep], res['asr_cancer_incidence'][keep])
    axes[2].set_title('Age-standardized cancer incidence')

    fig.tight_layout()
    fig.show()
    return fig


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    location = 'zambia'

    # Run calibration - usually on VMs
    if 'run_calibration' in to_run:
        filestem = ''
        sim, calib = run_calib(location=location, n_trials=n_trials, n_workers=n_workers,
                               do_save=do_save, do_plot=False, filestem=filestem)

    # Load the calibration, plot it, and save the best parameters -- usually locally
    if 'plot_calibration' in to_run:

        filestem = ''
        calib = load_calib(location=location, do_plot=True, save_pars=True, filestem=filestem)
        plot_extra_results(calib)

    T.toc('Done')
