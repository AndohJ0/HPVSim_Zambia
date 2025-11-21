"""
Helper functions for Zambia analysis
"""

# Standard imports
import numpy as np
import sciris as sc
import hpvsim as hpv
import pylab as pl
import pandas as pd
import time
import gc

#%% Settings and filepaths

# Debug switch
debug = 0  # Run with smaller population sizes and in serial
do_shrink = True  # Do not keep people when running sims (saves memory)

""" 1. make_sim --> Standard simulation creation function """

def make_sim(calib=False, calib_pars=None, debug=0, interventions=None, seed=1, end=None, analyzers=None,
             datafile=None, hiv_datafile=None, art_datafile=None, model_hiv=True):
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
        beta=0.16,
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
        model_hiv=model_hiv,
        hiv_pars={},
    )

    # Latency parameters (not modelling HPVlatency)
    # pars.hpv_control_prob = 0.0  # Probability that HPV is controlled latently vs. cleared
    # pars.hpv_reactivation = 0.025  # Probability of a latent infection reactivating

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
        pars=pars, interventions=interventions, rand_seed=seed, analyzers=analyzers,
        datafile=datafile, hiv_datafile=hiv_datafile, art_datafile=art_datafile
    )

    return sim

""" 2. run_sim --> Simulation running function """

def run_sim(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_datafile=None, art_datafile=None,
        location='zambia', model_hiv=True):

    dflocation = location.replace(' ', '_')

    # Make arguments
    if hiv_datafile is None:
        hiv_datafile = [f'data/{dflocation}_hiv_incidence_updated.csv',
                        f'data/{dflocation}_female_hiv_mortality_updated.csv',
                        f'data/{dflocation}_male_hiv_mortality_updated.csv']
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
        calib_pars=calib_pars,
        model_hiv=model_hiv
    )
    sim.label = f'Sim--{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    # Optionally save
    if do_save:
        sim.save(f'results/zambia.sim')

    return sim

""" 3. get_top_calibrated_pars --> Function to get the top calibrated parameter sets from the calibration results """
def get_top_calibrated_pars(calib, n=None):
    """Return the top-n calibrated parameter sets with metadata."""
    available = int(min(n, len(calib.df)))
    top_pars = []
    for i in range(available):
        trial_row = calib.df.iloc[i]
        pars = calib.trial_pars_to_sim_pars(which_pars=i)
        top_pars.append(dict(
            pars=pars,
            trial_index=int(trial_row['index']),
            mismatch=float(trial_row['mismatch']),
            rank=i + 1,
        ))
    return top_pars

    
""" 3. run_multi_sim --> Multi-simulation function for running multiple simulations  """

def run_multi_sim(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_datafile=None, art_datafile=None,
        n_runs=10, batch_size=None, top_pars=None, create_reduced=True, model_hiv=True):
    """Run multiple simulations with analyzers using optimized batch processing across top calibrated parameter sets.
    
    Optimized for large numbers of parameter sets (e.g., 100 parameter sets with 10 runs each).
    """
    
    # Normalize top_pars input
    if top_pars is None:
        top_pars = [{'pars': calib_pars, 'rank': None}]
    elif isinstance(top_pars, dict):
        top_pars = [top_pars]
    
    # Optimize batch_size: use smaller batches for many parameter sets to save memory
    if batch_size is None:
        if len(top_pars) > 50:
            batch_size = min(10, n_runs)  # Smaller batches for many parameter sets
        else:
            batch_size = min(25, n_runs)  # Standard batch size
    
    # Make arguments
    if hiv_datafile is None:
        hiv_datafile = ['data/zambia_hiv_incidence_updated.csv', 
                        'data/zambia_female_hiv_mortality_updated.csv',
                        'data/zambia_male_hiv_mortality_updated.csv']
    if art_datafile is None:
        art_datafile = ['data/zambia_art_coverage.csv']

    total_runs = n_runs * len(top_pars)
    print(f'Running {total_runs} simulations ({n_runs} per parameter set, {len(top_pars)} parameter sets)')
    print(f'Using batch size: {batch_size}')
    
    # Reduce verbose output for individual sims when running many parameter sets
    sim_verbose = 0.0 if len(top_pars) > 20 else verbose
    
    start_time = time.time()
    
    all_sims = []
    all_reduced = []
    total_completed = 0
    last_progress_print = 0
    progress_interval = max(1, total_runs // 50)  # Update progress every ~2% or at least every simulation
    
    for idx, cfg in enumerate(top_pars):
        cfg_pars = cfg.get('pars', calib_pars)
        cfg_rank = cfg.get('rank', idx + 1)
        cfg_label = cfg.get('label', f'top_{cfg_rank:02d}')
        cfg_mismatch = cfg.get('mismatch', None)
        
        # Show parameter set info only for small runs or every 10 sets for large runs
        if len(top_pars) <= 20:
            print(f'\n=== Parameter set {idx + 1}/{len(top_pars)}: {cfg_label} (rank {cfg_rank}) ===')
            if cfg_mismatch is not None:
                print(f'Mismatch: {cfg_mismatch:.6f}')
        elif (idx + 1) % 10 == 0 or idx == 0:
            print(f'\n=== Parameter set {idx + 1}/{len(top_pars)}: {cfg_label} (rank {cfg_rank}) ===')
        
        batch_num = 0
        for batch_start in range(0, n_runs, batch_size):
            batch_num += 1
            batch_end = min(batch_start + batch_size, n_runs)
            batch_runs = batch_end - batch_start
            
            # Only show batch details for small runs
            if len(top_pars) <= 20:
                print(f'  Batch {batch_num}: runs {batch_start+1}-{batch_end} ({batch_runs} simulations)')
            
            # Create fresh base sim for each batch to avoid memory accumulation
            base_sim = make_sim(
                debug=debug,
                seed=seed + idx * 10000 + batch_start,  # Use different seeds for each parameter set and batch
                end=end,
                hiv_datafile=hiv_datafile,
                art_datafile=art_datafile,
                analyzers=analyzers,
                interventions=interventions,
                calib_pars=cfg_pars,
                model_hiv=model_hiv
            )
            base_sim['verbose'] = sim_verbose  # Use reduced verbosity for individual sims
            
            # Create and run MultiSim for this batch
            msim = hpv.MultiSim(base_sim)
            msim.run(n_runs=batch_runs)
            
            # Store results from this batch and add metadata
            for sim in msim.sims:
                sim.rank = cfg_rank
                sim.top_label = cfg_label
                if cfg_mismatch is not None:
                    sim.mismatch = cfg_mismatch
                all_sims.append(sim)
            
            total_completed += batch_runs
            
            # Print total progress regularly
            if total_completed - last_progress_print >= progress_interval or total_completed == total_runs:
                elapsed = time.time() - start_time
                percent = (total_completed / total_runs) * 100
                rate = total_completed / elapsed if elapsed > 0 else 0
                remaining = (total_runs - total_completed) / rate if rate > 0 else 0
                print(f'  Total progress: {total_completed}/{total_runs} simulations ({percent:.1f}%) | '
                      f'Elapsed: {elapsed/60:.1f} min | Remaining: ~{remaining/60:.1f} min | '
                      f'Rate: {rate:.2f} sims/min')
                last_progress_print = total_completed
            
            # Force garbage collection to free memory
            del msim, base_sim
            gc.collect()
            
            if len(top_pars) <= 20:
                print(f'  Completed batch {batch_num} ({batch_runs} simulations)')
        
        # Create a reduced median sim per parameter set (optional, can be expensive for many sets)
        if create_reduced:
            try:
                final_base = make_sim(
                    debug=debug,
                    seed=seed + idx * 10000,
                    end=end,
                    hiv_datafile=hiv_datafile,
                    art_datafile=art_datafile,
                    analyzers=analyzers,
                    interventions=interventions,
                    calib_pars=cfg_pars,
                    model_hiv=model_hiv
                )
                final_msim = hpv.MultiSim(final_base)
                subset = [s for s in all_sims if getattr(s, 'rank', None) == cfg_rank]
                final_msim.sims = subset
                final_msim.median()
                reduced_sim = final_msim.base_sim
                reduced_sim.rank = cfg_rank
                reduced_sim.top_label = cfg_label
                if cfg_mismatch is not None:
                    reduced_sim.mismatch = cfg_mismatch
                all_reduced.append((final_msim, reduced_sim))
            except Exception as e:
                if len(top_pars) <= 20:
                    print(f'  Warning: Could not create reduced sim for rank {cfg_rank}: {e}')
    
    end_time = time.time()
    elapsed_total = end_time - start_time
    print(f'\nCompleted all {total_runs} simulations in {elapsed_total/60:.1f} minutes ({elapsed_total:.2f} seconds)')
    print(f'Average: {elapsed_total/total_runs:.2f} seconds per simulation')
    
    return all_sims, all_reduced


def _scale_art_datafiles(art_datafile, scale):
    """Return list of art coverage files scaled by `scale` (writes new CSVs if needed)."""
    if scale == 1.0 or not art_datafile:
        return art_datafile

    art_files = art_datafile if isinstance(art_datafile, (list, tuple)) else [art_datafile]
    scaled_files = []
    for path in art_files:
        try:
            df = pd.read_csv(path)
            numeric_cols = [
                col for col in df.columns
                if df[col].dtype.kind in 'fi' and ('ART' in col.upper() or 'COVERAGE' in col.upper())
            ]
            if numeric_cols:
                df[numeric_cols] = df[numeric_cols] * scale
            suffix = f'_scaled_{scale}'.replace('.', 'p')
            out_path = path.replace('.csv', f'{suffix}.csv')
            df.to_csv(out_path, index=False)
            scaled_files.append(out_path)
        except Exception:
            scaled_files.append(path)
    return scaled_files


def run_multi_sim_optimized_art(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_datafile=None, art_datafile=None,
        n_runs=10, batch_size=None, top_pars=None, create_reduced=True,
        art_coverage_scale=1.0, model_hiv=True):
    """Wrapper around run_multi_sim with optional ART coverage scaling and HIV toggling."""
    if hiv_datafile is None:
        hiv_datafile = ['data/zambia_hiv_incidence_updated.csv',
                        'data/zambia_female_hiv_mortality_updated.csv',
                        'data/zambia_male_hiv_mortality_updated.csv']
    if art_datafile is None:
        art_datafile = ['data/zambia_art_coverage.csv']

    scaled_art_files = _scale_art_datafiles(art_datafile, art_coverage_scale)

    return run_multi_sim(
        analyzers=analyzers,
        interventions=interventions,
        debug=debug,
        seed=seed,
        verbose=verbose,
        do_save=do_save,
        end=end,
        calib_pars=calib_pars,
        hiv_datafile=hiv_datafile,
        art_datafile=scaled_art_files,
        n_runs=n_runs,
        batch_size=batch_size,
        top_pars=top_pars,
        create_reduced=create_reduced,
        model_hiv=model_hiv,
    )

""" 4. q25_func and q75_func --> Functions to calculate 25th and 75th percentiles """
def q25_func(data):
    """Calculate 25th percentile (IQR low) along axis 0."""
    return np.percentile(data, 25, axis=0)


def q75_func(data):
    """Calculate 75th percentile (IQR high) along axis 0."""
    return np.percentile(data, 75, axis=0)


DEFAULT_EXPORT_METRICS = [
    'cancers', 'cancers_with_hiv', 'cancers_no_hiv',
    'cancer_incidence', 'cancer_incidence_with_hiv', 'cancer_incidence_no_hiv'
]

""" 5. aggregate_metric_series --> Function to aggregate metric series across simulations """
def aggregate_metric_series(sims, metric, label_fmt='run_{rank:02d}'):
    """Return a DataFrame aggregating a single metric across sims."""
    metric_key = metric
    years = np.asarray(sims[0].results['year'])
    columns = {'year': years}

    collected = []
    for sim in sims:
        if metric_key not in sim.results:
            continue
        label = label_fmt.format(rank=int(getattr(sim, 'rank', 0) or int(sim.label.split()[1])))
        series = np.asarray(sim.results[metric_key])
        columns[label] = series
        collected.append(series)

    if not collected:
        return None

    stacked = np.vstack(collected)
    columns['mean'] = stacked.mean(axis=0)
    columns['median'] = np.median(stacked, axis=0)
    columns['q25'] = q25_func(stacked)
    columns['q75'] = q75_func(stacked)
    columns['min'] = stacked.min(axis=0)
    columns['max'] = stacked.max(axis=0)

    return pd.DataFrame(columns)


""" 6. aggregate_and_export --> Function to aggregate metrics across simulations and export to Excel """
def aggregate_and_export(sims, location, metrics=None, save=True, outfile_suffix=''):
    """Aggregate selected metrics across sims and optionally export to Excel."""
    if not sims:
        return {}

    if metrics is None:
        metrics = DEFAULT_EXPORT_METRICS

    aggregated = {}
    for metric in metrics:
        df = aggregate_metric_series(sims, metric)
        if df is not None:
            aggregated[metric] = df

    if aggregated and save:
        suffix = outfile_suffix or ''
        outfile = f'results/{location}_top_calibrated_aggregated{suffix}.xlsx'
        with pd.ExcelWriter(outfile, engine='xlsxwriter') as writer:
            for metric, df in aggregated.items():
                df.to_excel(writer, sheet_name=metric, index=False)
        print(f'Aggregated metrics saved to {outfile}')

    return aggregated


""" 7b. export_raw_sim_series --> Export raw timeseries for each simulation """
def export_raw_sim_series(sims, location, metrics=None, save_csv=True, save_xlsx=True,
                          csv_path=None, xlsx_path=None):
    """Export per-simulation metric timeseries so individual sims can be inspected."""
    if not sims or not (save_csv or save_xlsx):
        return {}

    if metrics is None:
        metrics = DEFAULT_EXPORT_METRICS

    rows = []
    for sim_idx, sim in enumerate(sims, start=1):
        years = np.asarray(sim.results['year'])
        sim_meta = dict(
            sim_index=sim_idx,
            rank=getattr(sim, 'rank', None),
            top_label=getattr(sim, 'top_label', getattr(sim, 'label', f'sim_{sim_idx}')),
            mismatch=getattr(sim, 'mismatch', None),
        )
        for metric in metrics:
            if metric not in sim.results:
                continue
            values = np.asarray(sim.results[metric])
            for year, value in zip(years, values):
                rows.append({
                    **sim_meta,
                    'metric': metric,
                    'year': year,
                    'value': float(value),
                })

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    outputs = {}
    if save_csv:
        csv_file = csv_path or f'results/{location}_top_calibrated_all_sims.csv'
        df.to_csv(csv_file, index=False)
        outputs['csv'] = csv_file
        print(f'All raw sims exported to {csv_file}')
    if save_xlsx:
        xlsx_file = xlsx_path or f'results/{location}_top_calibrated_all_sims.xlsx'
        df.to_excel(xlsx_file, index=False)
        outputs['xlsx'] = xlsx_file
        print(f'All raw sims exported to {xlsx_file}')

    return outputs

""" 7. create_age_analyzer and aggregate_analyzer_results --> Functions for analyzer creation and aggregation """

def create_age_analyzer():
    """Create age-stratified analyzer for cancer and HIV results."""
    return hpv.age_results(
        result_args=sc.objdict( 
            cancers_no_hiv=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancers_with_hiv=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancers=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancer_incidence_no_hiv=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancer_incidence_with_hiv=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancer_incidence=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
            cancer_hiv_rate_ratios=sc.objdict(years=2020, edges=np.array([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,100.])),
        )
    )

def aggregate_analyzer_results(sims):
    """Aggregate analyzer results across simulations into DataFrames with statistics."""
    print("Aggregating analyzer results across all simulations...")
    
    all_analyzer_dfs = []
    
    for i, sim in enumerate(sims):
        if i % 10 == 0:
            print(f"Processing simulation {i+1}/{len(sims)}")
        
        try:
            analyzer = sim.get_analyzer()
        except (ValueError, AttributeError):
            print(f"Warning: No analyzer found for simulation {i+1}, skipping...")
            continue
        
        if analyzer is not None:
            year_key = np.int64(2020)
            # Get rank and mismatch if available
            rank = getattr(sim, 'rank', None)
            mismatch = getattr(sim, 'mismatch', None)
            df = pd.DataFrame({
                'bins': analyzer.results['cancers']['bins'],
                'cancers': analyzer.results['cancers'][year_key],
                'cancers_with_hiv': analyzer.results['cancers_with_hiv'][year_key],
                'cancers_no_hiv': analyzer.results['cancers_no_hiv'][year_key],
                'cancer_incidence': analyzer.results['cancer_incidence'][year_key],
                'cancer_incidence_with_hiv': analyzer.results['cancer_incidence_with_hiv'][year_key],
                'cancer_incidence_no_hiv': analyzer.results['cancer_incidence_no_hiv'][year_key],
                'cancer_rate_ratio': analyzer.results['cancer_hiv_rate_ratios'][year_key],
                'simulation_id': i + 1,
                'rank': rank if rank is not None else np.nan,
                'mismatch': mismatch if mismatch is not None else np.nan
            })
            all_analyzer_dfs.append(df)
    
    if not all_analyzer_dfs:
        raise ValueError("No analyzer results found in any simulations")
    
    combined_df = pd.concat(all_analyzer_dfs, ignore_index=True)
    
    # Calculate aggregate statistics with quartiles
    aggregate_stats = combined_df.groupby('bins').agg({
        'cancers': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancers_with_hiv': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancers_no_hiv': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancer_incidence': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancer_incidence_with_hiv': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancer_incidence_no_hiv': ['median', q25_func, q75_func, 'mean', 'min', 'max'],
        'cancer_rate_ratio': ['median', q25_func, q75_func, 'mean', 'min', 'max']
    }).round(2)
    
    aggregate_stats.columns = ['_'.join(col).strip() for col in aggregate_stats.columns]
    aggregate_stats = aggregate_stats.reset_index()
    
    return combined_df, aggregate_stats


""" 8. run_multi_sim_with_analyzers --> Function to run multiple simulations with analyzers """

def run_multi_sim_with_analyzers(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_datafile=None, art_datafile=None,
        n_runs=100, batch_size=None, top_pars=None, create_reduced=True):
    """Run multiple simulations with analyzers using optimized batch processing.
    
    Optimized for large numbers of parameter sets (e.g., 100 parameter sets with 10 runs each).
    When top_pars is provided, automatically uses optimized batch processing.
    """
    
    # If top_pars is provided, use run_multi_sim which handles multiple parameter sets
    if top_pars is not None:
        return run_multi_sim(
            analyzers=analyzers,
            interventions=interventions,
            debug=debug,
            seed=seed,
            verbose=verbose,
            do_save=do_save,
            end=end,
            calib_pars=calib_pars,
            hiv_datafile=hiv_datafile,
            art_datafile=art_datafile,
            n_runs=n_runs,
            batch_size=batch_size,
            top_pars=top_pars,
            create_reduced=create_reduced
        )
    
    # Otherwise, run with single parameter set (original behavior)
    # Make arguments
    if hiv_datafile is None:
        hiv_datafile = ['data/zambia_hiv_incidence_updated.csv', 
                        'data/zambia_female_hiv_mortality_updated.csv',
                        'data/zambia_male_hiv_mortality_updated.csv']
    if art_datafile is None:
        art_datafile = ['data/zambia_art_coverage.csv']
    
    # Set default batch_size if not provided
    if batch_size is None:
        batch_size = min(25, n_runs)

    print(f'Running {n_runs} simulations with analyzers in batches of {batch_size}...')
    start_time = time.time()
    
    all_sims = []
    batch_num = 0
    
    for batch_start in range(0, n_runs, batch_size):
        batch_num += 1
        batch_end = min(batch_start + batch_size, n_runs)
        batch_runs = batch_end - batch_start
        
        print(f'Processing batch {batch_num}: runs {batch_start+1}-{batch_end} ({batch_runs} simulations)')
        
        # Create fresh base sim for each batch to avoid memory accumulation
        base_sim = make_sim(
            debug=debug,
            seed=seed + batch_start,  # Use different seeds for each batch
            end=end,
            hiv_datafile=hiv_datafile,
            art_datafile=art_datafile,
            analyzers=analyzers,
            interventions=interventions,
            calib_pars=calib_pars
        )
        base_sim['verbose'] = verbose
        
        # Create and run MultiSim for this batch
        msim = hpv.MultiSim(base_sim)
        msim.run(n_runs=batch_runs)
        
        # Store results from this batch
        all_sims.extend(msim.sims)
        
        # Force garbage collection to free memory
        del msim, base_sim
        gc.collect()
        
        print(f'Completed batch {batch_num} ({batch_runs} simulations)')
    
    end_time = time.time()
    print(f'Completed all {n_runs} simulations in {end_time - start_time:.2f} seconds')
    
    return all_sims

""" 9. run_single_par_set --> Function to run simulations for a single parameter set (for SLURM job arrays) """

def run_single_par_set(par_set, analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
                       do_save=False, end=2020, hiv_datafile=None, art_datafile=None,
                       n_runs=100, batch_size=25, location='zambia', output_dir='results'):
    """Run simulations for a single parameter set. Used for SLURM job array parallelization."""
    import os
    
    rank = par_set.get('rank', 'unknown')
    mismatch = par_set.get('mismatch', None)
    calib_pars = par_set.get('pars', None)
    
    print(f"\n{'='*80}")
    print(f"Running simulations for parameter set rank {rank}")
    if mismatch is not None:
        print(f"Mismatch: {mismatch:.6f}")
    print(f"{'='*80}\n")
    
    # Run simulations with this parameter set
    sims = run_multi_sim_with_analyzers(
        analyzers=analyzers,
        interventions=interventions,
        debug=debug,
        seed=seed + rank * 10000,  # Different seed per parameter set
        verbose=verbose,
        do_save=False,  # We'll save separately
        end=end,
        calib_pars=calib_pars,
        hiv_datafile=hiv_datafile,
        art_datafile=art_datafile,
        n_runs=n_runs,
        batch_size=batch_size
    )
    
    # Add metadata to simulations
    for sim in sims:
        sim.rank = rank
        sim.mismatch = mismatch
        sim.top_label = f'top_{rank:02d}'
    
    # Save results for this parameter set
    if do_save:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual simulations
        sims_file = f'{output_dir}/{location}_rank{rank:02d}_sims.obj'
        sc.saveobj(sims_file, sims)
        print(f'Saved {len(sims)} simulations to {sims_file}')
        
        # Aggregate and save analyzer results if available
        try:
            combined_df, aggregate_stats = aggregate_analyzer_results(sims)
            combined_df.to_excel(f'{output_dir}/{location}_rank{rank:02d}_analyzer_all_sims_2020.xlsx', index=False)
            aggregate_stats.to_excel(f'{output_dir}/{location}_rank{rank:02d}_analyzer_aggregated_2020.xlsx', index=False)
            print(f'Saved analyzer results for rank {rank}')
        except Exception as e:
            print(f'Warning: Could not aggregate analyzer results for rank {rank}: {e}')
    
    return sims"""
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
