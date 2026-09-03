"""
Run simulations with top calibrated parameters for Zambia
including analyzers and plotting the results
"""

# Standard imports
import sciris as sc

# Helper functions from this repository
from run_functions import (
    get_top_calibrated_pars, run_multi_sim_optimized_art, run_multi_sim_with_analyzers,
    export_raw_sim_series, aggregate_and_export, create_age_analyzer, aggregate_analyzer_results,
)

# Save settings
do_save = True
art_coverage_scale = 0  # ART coverage multiplier: 0 = counterfactual without ART, 1 = actual coverage
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
    calib_file = f'results/{location}_calib.obj'
    try:
        calib = sc.loadobj(calib_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'{calib_file} not found -- run run_calibration.py first to produce it.'
        )

    if 'sim_with_top_pars' in to_run:
        top_pars = get_top_calibrated_pars(calib, n=100)  # Use the 100 best-fitting parameter sets
        sims, _ = run_multi_sim_optimized_art(
            top_pars=top_pars,
            end=2025,
            n_runs=100,  # Runs (different seeds) per parameter set
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
        sims, _ = run_multi_sim_with_analyzers(top_pars=top_pars, end=2025, analyzers=az1, n_runs=100,
                                               model_hiv=include_hiv)
        # Aggregate and export standard metrics
        aggregate_and_export(sims, location, save=do_save)
        # Aggregate analyzer results
        combined_df, aggregate_stats = aggregate_analyzer_results(sims)
        if do_save:
            combined_df.to_excel(f'results/{location}_top_calibrated_analyzer_results_all_sims_2020.xlsx', index=False)
            aggregate_stats.to_excel(f'results/{location}_top_calibrated_analyzer_results_aggregated_2020.xlsx', index=False)
            print(f'Analyzer results saved to results/{location}_top_calibrated_analyzer_results_*.xlsx')

    T.toc('Done')
