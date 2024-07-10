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

# Imports from this repository
import run_sim as rs


# CONFIGURATIONS TO BE SET BY USERS BEFORE RUNNING
to_run = [
    'run_calibration',  # Make sure this is uncommented if you want to _run_ the calibrations (usually on VMs)
    # 'plot_calibration',  # Make sure this is uncommented if you want to _plot_ the calibrations (usually locally)
]
debug = False  # If True, this will do smaller runs that can be run locally for debugging
do_save = True

# Run settings for calibration (dependent on debug)
n_trials = [5000, 10][debug]  # How many trials to run for calibration
n_workers = [40, 1][debug]  # How many cores to use
storage = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations


########################################################################
# Run calibration
########################################################################
def make_priors():
    return


def run_calib(location=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):
    dflocation = location.replace(" ", "_")
    hiv_datafile = [f'data/{dflocation}_hiv_incidence.csv',
                    f'data/{dflocation}_female_hiv_mortality.csv',
                    f'data/{dflocation}_male_hiv_mortality.csv']
    art_datafile = [f'data/{dflocation}_art_coverage_by_age_males.csv',
                    f'data/{dflocation}_art_coverage_by_age_females.csv']

    sim = rs.make_sim(hiv_datafile=hiv_datafile, art_datafile=art_datafile, calib=True)

    datafiles = [
        f'data/{dflocation}_cancer_cases.csv',  # Globocan
        f'data/{dflocation}_asr_cancer_incidence.csv',
    ]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.05, 0.02, 0.5, 0.02],
        own_imm_hr=[0.5, 0.25, 1, 0.05],
        age_risk=dict(risk=[3.2, 1, 4, 0.1],
                      age=[38, 30, 45, 1]),
        hpv_control_prob=[0, 0, 1, 0.25],
        hpv_reactivation=[0.025, 0, 0.1, 0.025]
    )

    sexual_behavior_pars = dict(
        m_cross_layer=[0.3, 0.1, 0.7, 0.05],
        m_partners=dict(
            c=dict(par1=[0.5, 0.1, 0.6, 0.05])
        ),
        f_cross_layer=[0.4, 0.05, 0.7, 0.05],
        f_partners=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.05])
        )
    )

    calib_pars = sc.mergedicts(calib_pars, sexual_behavior_pars)

    hiv_pars = dict(
        rel_sus=dict(
            lt200=[2.25, 2, 5, 0.25],
            gt200=[2.25, 2, 4, 0.25]
        ),
        rel_sev=dict(
            lt200=[1.5, 1.25, 5, 0.25],
            gt200=[1.5, 1.25, 3, 0.25]
        ),
        rel_reactivation_prob=[3, 2, 5, 0.5]
    )

    # Save some extra sim results
    extra_sim_result_keys = ['cancers', 'cancers_with_hiv', 'cancers_no_hiv',
                             'cancers_by_age_with_hiv', 'cancers_by_age_no_hiv',
                             'asr_cancer_incidence', 'cancer_incidence_by_age_with_hiv',
                             'cancer_incidence_by_age_no_hiv']

    calib = hpv.Calibration(sim, calib_pars=calib_pars, genotype_pars=genotype_pars,
                            hiv_pars=hiv_pars,
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
    calib = sc.load(f'results/{filename}.obj')
    if do_plot:
        sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
        sc.options(font='Libertinus Sans')
        fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
        fig.suptitle(f'Calibration results, {location.capitalize()}')
        fig.tight_layout()
        fig.savefig(f'figures/{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        sc.save(f'results/{location}_pars{filestem}.obj', calib_pars)

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

    T.toc('Done')
