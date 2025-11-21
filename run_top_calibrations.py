"""
Run simulations with top calibrated parameters for Zambia
including analyzers and plotting the results
"""

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pylab as pl
import pandas as pd
import time 

#import helper functions
from run_functions import *

# Save settings
do_save = True
save_plots = True
art_coverage_scale = 0  # Set to 0 for counterfactual without ART, etc.
include_hiv = True  # Toggle HIV co-infection dynamics


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()

    # Make a list of what to run, comment out anything you don't want to run
    to_run = [
        'sim_with_top_pars',
        #'sim_with_top_pars_and_analyzers',
    ]

    location = 'zambia'
    calib = sc.loadobj(f'results/{location}_calib.obj')

    if 'sim_with_top_pars' in to_run:
        top_pars = get_top_calibrated_pars(calib, n=100)
        sims, _ = run_multi_sim_optimized_art(
            top_pars=top_pars,
            end=2025,
            n_runs=100,
            batch_size=25,
            art_coverage_scale=art_coverage_scale,
            model_hiv=include_hiv,
        )

        if do_save:
            scale_tag = str(art_coverage_scale).replace('.', 'p')
            suffix_parts = ['with_hiv' if include_hiv else 'no_hiv', f'artscale_{scale_tag}']
            scenario_suffix = '_' + '_'.join(suffix_parts)
            export_raw_sim_series(
                sims,
                location,
                save_csv=True,
                save_xlsx=False,
                xlsx_path=f'results/{location}_top_calibrated_all_sims{scenario_suffix}.xlsx'
            )

        agg_suffix = scenario_suffix if do_save else ''
        aggregate_and_export(
            sims,
            location,
            save=do_save,
            outfile_suffix=agg_suffix
        )

    if 'sim_with_top_pars_and_analyzers' in to_run:
        top_pars = get_top_calibrated_pars(calib, n=100)
        # Print parameter set information
        print("\n" + "="*80)
        print("Running simulations with top calibrated parameter sets:")
        print("="*80)
        for par_set in top_pars:
            print(f"  Rank {par_set['rank']:2d}: Mismatch = {par_set['mismatch']:.6f}, Trial Index = {par_set['trial_index']}")
        print("="*80 + "\n")
        # Create age-stratified analyzer
        az1 = create_age_analyzer()
        # Run simulations with analyzers
        sims, _ = run_multi_sim_with_analyzers(top_pars=top_pars, end=2025, analyzers=az1, n_runs=100)
        # Aggregate and export standard metrics
        aggregate_and_export(sims, location, save=do_save)
        # Aggregate analyzer results
        combined_df, aggregate_stats = aggregate_analyzer_results(sims)
        if do_save:
            combined_df.to_excel(f'results/{location}_top_calibrated_analyzer_results_all_sims_2020.xlsx', index=False)
            aggregate_stats.to_excel(f'results/{location}_top_calibrated_analyzer_results_aggregated_2020.xlsx', index=False)
            print(f'Analyzer results saved to results/{location}_top_calibrated_analyzer_results_*.xlsx')

    T.toc('Done')
