"""
Define an HPVsim simulation for Zambia
"""

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pylab as pl

#%% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


#%% Simulation creation functions
def make_sim(calib=False, calib_pars=None, debug=0, analyzers=None, interventions=None, seed=1, end=None,
             datafile=None, hiv_datafile=None, art_datafile=None):
    """"
    Define parameters, analyzers, and interventions for the simulation
    """
    if end is None:
        end = 2100
    if calib:
        end = 2020

    pars = sc.objdict(
        n_agents=[10e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=end,
        genotypes=[16, 18, 'hi5', 'ohr'],
        location='zambia',
        init_hpv_dist=dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=.1),
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=100,
        verbose=0.0,
        rand_seed=seed,
        model_hiv=True,
        hiv_pars=dict(),
    )

    # Latency parameters
    pars.hpv_control_prob = 0.0  # Probability that HPV is controlled latently vs. cleared
    pars.hpv_reactivation = 0.025  # Probability of a latent infection reactivating

    # Sexual behavior parameters
    # Debut: derived by fitting to 2018 DHS
    # Women:
    #           Age:   15,   18,   20,   22,   25
    #   Prop_active: 17.1, 68.6, 84.4, 91.5, 94.9
    # Men:
    #           Age:   15,   18,   20,   22,   25
    #   Prop_active: 10.5, 42.8, 66.4, 81.7, 91.9
    # For fitting, see https://www.researchsquare.com/article/rs-3074559/v1
    pars.debut = dict(
        f=dict(dist='lognormal', par1=16.69, par2=1.78),
        m=dict(dist='lognormal', par1=18.65, par2=3.06),
    )

    # Participation in marital and casual relationships
    # Derived to fit 2018 DHS data
    # For fitting, see https://www.researchsquare.com/article/rs-3074559/v1
    pars.layer_probs = dict(
        m=np.array([
            # Share of people of each age who are married
            [0, 5,    10,       15,      20,     25,      30,     35,      40,     45,    50,   55,   60,   65,    70,    75],
            [0, 0, 0.009,   0.1314,  0.4734,  0.621,   0.675,  0.693,  0.6516, 0.6174,  0.45, 0.27, 0.18, 0.09, 0.045, 0.009],  # Females
            [0, 0,  0.01,    0.146,   0.526,   0.69,    0.75,   0.77,   0.724,  0.686,   0.5,  0.3,  0.2,  0.1,  0.05,  0.01]]  # Males
        ),
        c=np.array([
            # Share of people of each age in casual partnerships
            [0, 5,  10,  15,  20,  25,  30,  35,  40,   45,   50,   55,   60,   65,   70,   75],
            [0, 0, 0.1, 0.3, 0.3, 0.3, 0.3, 0.5, 0.6,  0.5,  0.4,  0.1, 0.01, 0.01, 0.01, 0.01],
            [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8,  0.6,  0.2,  0.1, 0.05, 0.02, 0.02, 0.02]]
        ),
    )

    pars.m_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson1', par1=0.2),
    )
    pars.f_partners = dict(
        m=dict(dist='poisson1', par1=0.01),
        c=dict(dist='poisson1', par1=0.2),
    )

    # HIV parameters
    pars.hiv_pars['art_failure_prob'] = 0.1

    # If calibration parameters have been supplied, use them here
    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    # Create the sim
    sim = hpv.Sim(
        pars=pars, interventions=interventions, analyzers=analyzers, rand_seed=seed,
        datafile=datafile, hiv_datafile=hiv_datafile, art_datafile=art_datafile
    )

    return sim


#%% Simulation running functions
def run_sim(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.2,
        do_save=False, end=2020, calib_pars=None, hiv_datafile=None, art_datafile=None):

    dflocation = location.replace(' ', '_')
    # Make arguments
    if hiv_datafile is None:
        hiv_datafile = [f'data/{dflocation}_hiv_incidence.csv',
                        f'data/{dflocation}_female_hiv_mortality.csv',
                        f'data/{dflocation}_male_hiv_mortality.csv']
    if art_datafile is None:
        art_datafile = [f'data/{dflocation}_art_coverage.csv']

    # Make sim
    sim = make_sim(
        debug=debug,
        seed=seed,
        end=end,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile,
        analyzers=analyzers,
        interventions=interventions,
        calib_pars=calib_pars
    )
    sim.label = f'Sim--{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    # Optinally save
    if do_save:
        sim.save(f'results/zambia.sim')

    return sim


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Make a list of what to run, comment out anything you don't want to run
    to_run = [
        'run_single',
        # 'run_scenario',
    ]

    location = 'zambia'
    calib_pars = None  #sc.loadobj(f'results/{location}_pars_nov06.obj')

    # Run and plot a single simulation
    # Takes <1min to run
    if 'run_single' in to_run:
        sim = run_sim(calib_pars=calib_pars, end=2020, debug=debug)  # Run the simulation
        sim.plot()  # Plot the simulation
 
    # Example of how to run a scenario with and without vaccination
    # Takes ~2min to run
    if 'run_scenario' in to_run:
        routine_vx = hpv.routine_vx(product='bivalent', age_range=[9, 10], prob=0.9, start_year=2025)
        sim_baseline = make_sim(calib_pars=calib_pars, end=2060)
        sim_scenario = make_sim(calib_pars=calib_pars, end=2060, interventions=routine_vx)
        msim = hpv.MultiSim(sims=[sim_baseline, sim_scenario])  # Make a multisim for running in parallel
        msim.run(verbose=0.1)

        # Now plot cancers with & without vaccination
        pl.figure()
        res0 = msim.sims[0].results
        res1 = msim.sims[1].results
        pl.plot(res0['year'][60:], res0['cancer_incidence'][60:], label='No vaccination')
        pl.plot(res0['year'][60:], res1['cancer_incidence'][60:], color='r', label='With vaccination')
        pl.legend()
        pl.title('Cancer incidence')
        pl.show()

    # To run more complex scenarios, you may want to set them up in a separate file

    T.toc('Done')  # Print out how long the run took
