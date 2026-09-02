"""
Define an HPVsim simulation for Zambia
"""

# Standard imports
import sciris as sc
import starsim as ss
import pylab as pl
import pandas as pd

# make_sim and run_sim live in run_functions so there is a single definition of
# the Zambia sim; this module is the script wrapper around them.
import hpvsim as hpv
from run_functions import make_sim, run_sim

#%% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Make a list of what to run, comment out anything you don't want to run
    to_run = [
        'run_single',
        # 'run_scenario',
    ]

    location = 'zambia'
    calib_pars = None  # sc.loadobj(f'results/{location}_pars.obj')

    # Run and plot a single simulation
    # Takes <1min to run
    if 'run_single' in to_run:
        sim = run_sim(calib_pars=calib_pars, end=2020, debug=debug)  # Run the simulation
        sim.to_excel('zambia_sim.xlsx')
        pd.read_excel('zambia_sim.xlsx').to_csv('zambia_sim.csv', index=False)
        sim.plot()  # Plot the simulation

    # Example of how to run a scenario with and without vaccination
    # Takes ~2min to run
    if 'run_scenario' in to_run:
        routine_vx = hpv.routine_vx(product='bivalent', age_range=[9, 10], prob=0.9, start_year=2025)
        sim_baseline = make_sim(calib_pars=calib_pars, stop=2060)
        sim_scenario = make_sim(calib_pars=calib_pars, stop=2060, interventions=routine_vx)
        msim = ss.MultiSim(sims=[sim_baseline, sim_scenario])  # Make a multisim for running in parallel
        msim.run(verbose=0.1)

        # Now plot cancers with & without vaccination
        pl.figure()
        years = msim.sims[0].results.timevec.years
        res0 = msim.sims[0].results.all_hpv
        res1 = msim.sims[1].results.all_hpv
        pl.plot(years[60:], res0['asr_cancer_incidence'][60:], label='No vaccination')
        pl.plot(years[60:], res1['asr_cancer_incidence'][60:], color='r', label='With vaccination')
        pl.legend()
        pl.title('Age-standardized cancer incidence')
        pl.show()

    # To run more complex scenarios, you may want to set them up in a separate file

    T.toc('Done')  # Print out how long the run took
