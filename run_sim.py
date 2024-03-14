'''
Define an HPVsim simulation for Zambia
'''

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pandas as pd
import seaborn as sns
import pylab as pl

# Imports from this repository
import behavior_inputs as bi

#%% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

# Save settings
do_save = True
save_plots = True


#%% Simulation creation functions
def make_sim(calib_pars=None, debug=0, analyzers=None, interventions=None, datafile=None, seed=1, end=2020):
    ''' Define parameters, analyzers, and interventions for the simulation -- not the sim itself '''

    pars = dict(
        n_agents       = [10e3,1e3][debug],
        dt             = [0.25,1.0][debug],
        start          = [1960,1980][debug],
        end            = end,
        network        = 'default',
        genotypes      = [16, 18, 'hi5', 'ohr'],
        location       = 'ethiopia',
        debut          = dict(f=dict(dist='lognormal', par1=16.8, par2=2.4),
                              m=dict(dist='lognormal', par1=22.5, par2=3.6)),
        mixing         = bi.default_mixing,
        layer_probs    = bi.default_layer_probs,
        partners       = bi.default_partners,
        init_hpv_dist  = dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=.1),
        init_hpv_prev  = {
            'age_brackets'  : np.array([  12,   17,   24,   34,  44,   64,    80, 150]),
            'm'             : np.array([ 0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f'             : np.array([ 0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio = 100,
        verbose        = 0.0,
    )

    # If calibration parameters have been supplied, use them here
    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    # Create the sim
    sim = hpv.Sim(pars=pars, interventions=interventions, analyzers=analyzers, datafile=datafile, rand_seed=seed)

    return sim


#%% Simulation running functions
def run_sim(calib_pars=None, debug=0, analyzers=None, interventions=None,
            datafile=None, seed=1, verbose=.1, do_save=False, end=2020):

    # Make sim
    sim = make_sim(
        debug=debug,
        seed=seed,
        end=end,
        datafile=datafile,
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


def run_sims(parsets=None, debug=False, verbose=-1, analyzers=None, save_results=True, **kwargs):
    ''' Run multiple simulations with different calibration parameter sets in parallel '''

    kwargs = sc.mergedicts(dict(debug=debug, verbose=verbose, analyzers=analyzers), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(calib_pars=parsets), kwargs=kwargs, serial=debug, die=True)
    msim = hpv.MultiSim(simlist)
    msim.reduce()
    if save_results:
        sc.saveobj(f'msim_zambia.obj', msim.results)

    return msim


#%% Run as a script
if __name__ == '__main__':

    # Make a list of what to run, comment out anything you don't want to run
    to_run = [
        'run_single',
        # 'run_scenario',
    ]

    T = sc.timer()  # Start a timer

    calib_pars = sc.loadobj('results/zambia_pars_nov06.obj')  # Load some parameters from a previous calibration

    # Run and plot a single simulation
    # Takes <1min to run
    if 'run_single' in to_run:
        sim = run_sim(calib_pars=calib_pars)  # Run the simulation
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

