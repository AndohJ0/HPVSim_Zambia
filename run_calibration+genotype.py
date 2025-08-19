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
import sciris as sc
import hpvsim as hpv
import pylab as pl
import pandas as pd

# Imports from this repository
import run_sim as rs


# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    #'run_calibration',  # Make sure this is uncommented if you want to _run_ the calibrations (usually on VMs)
    'plot_calibration',  # Make sure this is uncommented if you want to _plot_ the calibrations (usually locally)
]
debug = False  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

# Run settings for calibration (dependent on debug)
n_trials = [3000, 5][debug]  # How many trials to run for calibration
n_workers = [20, 1][debug]  # How many cores to use
storage = "sqlite:///hpvsim.db"  # Storage for calibrations


########################################################################
# Run calibration
########################################################################
def make_priors():
    return


def run_calib(location=None,calib_pars=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):
    dflocation = location.replace(" ", "_")
    hiv_datafile = [f'data/{dflocation}_hiv_incidence_updated.csv',
                    f'data/{dflocation}_female_hiv_mortality_updated.csv',
                    f'data/{dflocation}_male_hiv_mortality_updated.csv']
    art_datafile = [f'data/{dflocation}_art_coverage_by_age_males.csv',
                    f'data/{dflocation}_art_coverage_by_age_females.csv']

    sim = rs.make_sim(hiv_datafile=hiv_datafile, art_datafile=art_datafile, calib=True)

    datafiles = [
        f'/storage/homefs/ja22x644/HPVSim_zambia/data/zambia_cancer_cases.csv',  # Globocan
        f'/storage/homefs/ja22x644/HPVSim_zambia/data/zambia_asr_cancer_incidence.csv',
    ]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.15, 0.1, 0.5, 0.05], 
        own_imm_hr=[0.5, 0.25, 1, 0.05],
        age_risk=dict(risk=[3.2, 1, 4, 0.1],
                      age=[38, 30, 45, 1]),
        hpv_control_prob=[0, 0, 1, 0.25],
        hpv_reactivation=[0.025, 0, 0.1, 0.025],
        sev_dist=dict(par1=[1, 0.5, 1.5, 0.01])
    )

    sexual_behavior_pars = dict(
        m_cross_layer=[0.3, 0.1, 0.7, 0.05],
        m_partners=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        ),
        f_cross_layer=[0.1, 0.05, 0.5, 0.05],
        f_partners=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        )
    )

    calib_pars = sc.mergedicts(calib_pars, sexual_behavior_pars)

    genotype_pars = dict(
        hpv16=dict(
            dur_precin=dict(par1=[3, 1, 10, 0.5], par2=[9, 5, 15, 0.5]),
            cancer_fn=dict(transform_prob=[2e-3, 1e-3, 3e-3, 2e-4]),
            cin_fn=dict(k=[.35, .2, .4, 0.01]),
            dur_cin=dict(par1=[5, 4, 6, 0.5], par2=[20, 16, 24, 0.5]),
        ),
        hpv18=dict(
            dur_precin=dict(par1=[2.5, 1, 10, 0.5], par2=[9, 5, 15, 0.5]),
            cancer_fn=dict(transform_prob=[2e-3, 1e-3, 3e-3, 2e-4]),
            cin_fn=dict(k=[.4, .15, .35, 0.01]),
            dur_cin=dict(par1=[5, 4, 6, 0.5], par2=[20, 16, 24, 0.5]),
        ),
        hi5=dict(
            dur_precin=dict(par1=[2.5, 1, 10, 0.5], par2=[9, 5, 15, 0.5]),
            cancer_fn=dict(transform_prob=[1.5e-3, 0.5e-3, 2.5e-3, 2e-4]),
            cin_fn=dict(k=[.15, .1, .25, 0.01]),
            dur_cin=dict(par1=[4.5, 3.5, 5.5, 0.5], par2=[20, 16, 24, 0.5]),
        ),
        ohr=dict(
            dur_precin=dict(par1=[2.5, 1, 10, 0.5], par2=[9, 5, 15, 0.5]),
            cancer_fn=dict(transform_prob=[1.5e-3, 0.5e-3, 2.5e-3, 2e-4]),
            cin_fn=dict(k=[.15, .1, .25, 0.01]),
            dur_cin=dict(par1=[4.5, 3.5, 5.5, 0.5], par2=[20, 16, 24, 0.5]),
        ),
    )

    hiv_pars = dict(
        rel_sus=dict(
            lt200=[2.25, 2, 5, 0.25],
            gt200=[2.25, 2, 4, 0.25]
        ),
        rel_sev=dict(
            lt200=[1.5, 2, 5, 0.25],
            gt200=[1.5, 1.25, 3, 0.25]
        ),
        rel_reactivation_prob=[3, 2, 5, 0.5]
    )

    # Save some extra sim results
    extra_sim_result_keys = ['cancers', 'cancers_with_hiv', 'cancers_no_hiv',
                             'cancers_by_age_with_hiv', 'cancers_by_age_no_hiv',
                             'asr_cancer_incidence', 'cancer_incidence_by_age_with_hiv',
                             'cancer_incidence_by_age_no_hiv']

    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        hiv_pars=hiv_pars,
        genotype_pars=genotype_pars,
        name=f'{location}_calib',
        datafiles=datafiles,
        extra_sim_result_keys=extra_sim_result_keys,
        total_trials=n_trials, n_workers=n_workers,
        storage=storage
    )
    calib.calibrate()
    filename = f'{location}_calib{filestem}'
    if do_plot:
        calib.plot(do_save=True, fig_path=f'figures/{filename}.png')
    if do_save:
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib

########################################################################
# Load pre-run calibration
########################################################################
def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True, filestem=''):
    fnlocation = location.replace(' ', '_')
    filename = f'{fnlocation}_calib{filestem}'
    calib = sc.load(f'results/zambia_calib.obj')
    if do_plot:
        #sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
        #sc.options(font='Libertinus Sans')
        fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        sc.save(f'results/{filename}_pars.obj', calib_pars)

    return calib


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

        best_par_ind = calib.df.index[0]
        extra_sim_results = calib.extra_sim_results[best_par_ind]
        years = calib.sim.results['year']
        year_ind = sc.findinds(years, 1985)[0]

        fig, axes = pl.subplots(3, 1)
        axes[0].plot(years[year_ind:], extra_sim_results['cancers_with_hiv'][year_ind:], label='HIV+')
        axes[0].plot(years[year_ind:], extra_sim_results['cancers_no_hiv'][year_ind:], label='HIV-')
        axes[0].plot(years[year_ind:], extra_sim_results['cancers'][year_ind:], label='Total')
        axes[0].set_title(f'Cancers over time')
        axes[0].legend()
        axes[1].plot(calib.sim.pars['age_bin_edges'][:-1],
                     extra_sim_results['cancer_incidence_by_age_with_hiv'][:, -2], label='HIV+')
        axes[1].plot(calib.sim.pars['age_bin_edges'][:-1],
                     extra_sim_results['cancer_incidence_by_age_no_hiv'][:, -2],
                     label='HIV-')
        axes[1].legend()

        axes[2].plot(years[year_ind:], extra_sim_results['asr_cancer_incidence'][year_ind:])

        fig.show()
        fig.savefig(f'cancers_hiv_calib.png')

    T.toc('Done')
