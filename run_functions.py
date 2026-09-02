"""
Helper functions for Zambia analysis
"""

# Standard imports
import os
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv

from analyzers import CancerByAgeHIV
import pylab as pl
import pandas as pd
import time
import gc

#%% Settings and filepaths

""" 1. make_sim --> Standard simulation creation function """

LOCATION = 'zambia'


# --- v2.2.6 -> v2.3+ probability convention -------------------------------- #
# v2.3.0 reinterpreted layer_probs and the cross-layer probs from per-timestep
# to ANNUAL. Zambia's sexual-behaviour parameters were fitted under v2.2.6, so
# they are per-timestep and must be annualized or the network comes out ~dt
# times too sparse and the epidemic quietly dies. Note the exponent is 1/dt,
# not dt: at dt=0.25, p=0.1 -> 0.344, not 0.026.
def _to_annual_prob(p, dt):
    """Convert a per-timestep probability (v2.2.x) to an annual one (v2.3+)."""
    p = np.clip(p, 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def _layer_probs_to_annual(layer_probs, dt):
    """Annualize the female (row 1) and male (row 2) rows of a layer_probs array."""
    out = np.asarray(layer_probs, dtype=float).copy()
    out[1:3, :] = _to_annual_prob(out[1:3, :], dt)
    return out


# --- Zambia sexual behaviour, fitted to the 2018 DHS ----------------------- #
# For the fitting, see https://www.researchsquare.com/article/rs-3074559/v1
# Debut: women 17.1/68.6/84.4/91.5/94.9% sexually active by age 15/18/20/22/25;
# men 10.5/42.8/66.4/81.7/91.9%. See _behaviour_dists below.

# Share of people of each age in marital / casual partnerships. Rows are
# [age-bin lower bounds], [female], [male]. Per-timestep; annualized in make_sim.
_LAYER_PROBS_MARITAL = np.array([
    [0, 5,    10,      15,      20,     25,     30,     35,     40,     45,   50,   55,   60,   65,    70,    75],
    [0, 0, 0.009,  0.1314,  0.4734,  0.621,  0.675,  0.693, 0.6516, 0.6174, 0.45, 0.27, 0.18, 0.09, 0.045, 0.009],
    [0, 0,  0.01,   0.146,   0.526,   0.69,   0.75,   0.77,  0.724,  0.686,  0.5,  0.3,  0.2,  0.1,  0.05,  0.01]])
_LAYER_PROBS_CASUAL = np.array([
    [0, 5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,   60,   65,   70,   75],
    [0, 0, 0.1, 0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.5, 0.4, 0.1, 0.01, 0.01, 0.01, 0.01],
    [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.6, 0.2, 0.1, 0.05, 0.02, 0.02, 0.02]])

# v2's poisson1 added 1 to the draw; v3's SexualNetwork does that itself, so
# these are the bare Poisson rates.
_PARTNER_RATES = dict(m_marital=0.01, m_casual=0.2, f_marital=0.01, f_casual=0.2)

# Sexual debut, fitted to the 2018 DHS (mean, std in years).
_DEBUT = dict(f=(16.69, 1.78), m=(18.65, 3.06))


def _behaviour_dists():
    """Fresh distribution objects for the network parameters.

    Built per call, never shared: an ss.Dist carries RNG state, so reusing one
    instance across two sims raises DistSeedRepeatError under common random
    numbers.
    """
    pars = {f'{sex}_partners_{layer}': ss.poisson(lam=rate)
            for (key, rate) in _PARTNER_RATES.items()
            for sex, layer in [key.split('_')]}
    for sex, (mean, std) in _DEBUT.items():
        pars[f'debut_{sex}'] = ss.lognorm_ex(mean=mean, std=std)
    return pars


def _hiv_data(location=LOCATION):
    """Zambia HIV/ART inputs in the dict form hpv.Sim(hiv_data=) expects.

    v3 replaced v2's separate hiv_datafile/art_datafile lists with a single
    hiv_data=, either a folder of four fixed filenames or this dict. Zambia's
    files carry a `zambia_` prefix and there is no hiv_prevalence.csv, so the
    dict route is used; 'init_prev' is optional and omitted, leaving the HIV
    module to seed from the incidence curve alone.

    The two *_hiv_mortality_updated.csv files are deliberately unused: v3
    delegates HIV mortality to stisim, which models it endogenously from CD4
    progression rather than taking an imposed rate.
    """
    loc = location.replace(' ', '_')

    inc = pd.read_csv(f'data/{loc}_hiv_incidence_updated.csv').rename(
        columns={'Age': 'age', 'Year': 'year', 'Sex': 'sex', 'Incidence': 'incidence'})
    inc['sex'] = inc['sex'].astype(str).str.lower().str[0]
    inc = inc.astype({'age': int, 'year': int, 'incidence': float})

    # encoding: the by-age ART files begin with a BOM, which would otherwise
    # leave the first column named '﻿age'. dropna: they also carry ~500
    # trailing blank rows from the Excel export.
    frames = []
    for sex, fname in (('f', 'females'), ('m', 'males')):
        wide = pd.read_csv(f'data/{loc}_art_coverage_by_age_{fname}.csv',
                           encoding='utf-8-sig').dropna(subset=['age'])
        long = wide.melt(id_vars='age', var_name='year', value_name='coverage')
        long['sex'] = sex
        frames.append(long.astype({'age': int, 'year': int, 'coverage': float}))
    art = pd.concat(frames, ignore_index=True)

    return dict(incidence=inc[['age', 'sex', 'year', 'incidence']],
                art_coverage=art[['age', 'sex', 'year', 'coverage']])


def make_sim(calib=False, calib_pars=None, debug=0, interventions=None, seed=1, stop=None,
             analyzers=None, datafile=None, hiv_data=None, model_hiv=True, end=None):
    """Define parameters, analyzers, and interventions for the simulation.

    calib_pars is expected in v3 form (see hpv.route_pars): nested by scope,
    e.g. dict(beta=0.1, hiv=dict(rel_reactivation_lo=3.0),
    hpv16=dict(cin_fn=dict(k=0.35)), cross_immunity=dict(rel_sev=...)).
    Parameters recovered from a v2 calibration must be passed through
    v2_calib_pars_to_v3() first -- make_sim does not convert silently, since
    doing so would double-convert genuinely-v3 parameters.
    """
    if end is not None:  # v2 name; v3 uses stop=
        stop = end
    if stop is None:
        stop = 2100
    if calib:
        stop = 2020

    dt = [0.25, 1.0][debug]

    pars = sc.objdict(
        beta=0.16,
        # Zambia's v2 init_hpv_prev is byte-identical to v3's built-in default
        # age/sex curve (hpvsim/seeding.py), so it is simply dropped.
        ms_agent_ratio=100,  # Downsampling ratio between modeled and real-world agent counts
        verbose=0.0,
        # Sexual behaviour, annualized from the v2.2.6 per-timestep convention.
        layer_probs_marital=_layer_probs_to_annual(_LAYER_PROBS_MARITAL, dt),
        layer_probs_casual=_layer_probs_to_annual(_LAYER_PROBS_CASUAL, dt),
        **_behaviour_dists(),
    )

    # Latency: off by default (hpv_control_prob=0 makes it a no-op).
    pars.hpv_control_prob = 0.0
    pars.hpv_reactivation = 0.025

    # v2 carried art_failure_prob=0.1 as an hpvsim par; v3 delegates ART
    # suppression to stisim, where p_effective_art is its complement.
    hiv_pars = dict(p_effective_art=ss.bernoulli(p=0.9)) if model_hiv else None

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    if model_hiv and hiv_data is None:
        hiv_data = _hiv_data()

    return hpv.Sim(
        pars=pars,
        location=LOCATION,
        genotypes=[16, 18, 'hi5', 'ohr'],
        init_hpv_dist=dict(hpv16=0.4, hpv18=0.25, hi5=0.25, ohr=0.1),
        n_agents=[10e3, 1e3][debug],
        start=[1960, 1980][debug],
        stop=stop,
        dt=dt,
        rand_seed=seed,
        interventions=interventions,
        analyzers=analyzers,
        data=datafile,
        model_hiv=model_hiv or None,
        hiv_data=hiv_data if model_hiv else None,
        hiv_pars=hiv_pars,
    )


def v2_calib_pars_to_v3(v2_pars, dt=0.25):
    """Translate a v2.2.6 Zambia parameter dict into v3 form.

    Call this explicitly on anything recovered from the old .obj files (see
    results/v2_artefact_snapshot.json); make_sim does not do it implicitly.
    Renames to the v3 scoped/suffixed names, converts the cross-layer
    probabilities from per-timestep to annual, and drops v2 parameters that
    v3 has no equivalent for.
    """
    out = {}
    if 'beta' in v2_pars:
        out['beta'] = v2_pars['beta']
    for key in ('m_cross_layer', 'f_cross_layer'):
        if key in v2_pars:
            out[key] = float(_to_annual_prob(v2_pars[key], dt))
    for sex in 'mf':
        block = v2_pars.get(f'{sex}_partners')
        if block:
            for v2_layer, v3_layer in (('m', 'marital'), ('c', 'casual')):
                if v2_layer in block:
                    out[f'{sex}_partners_{v3_layer}'] = ss.poisson(lam=block[v2_layer]['par1'])
    # v2 sev_dist (individual biological severity) is v3's CrossImmunity.rel_sev.
    sev = v2_pars.get('sev_dist')
    if sev:
        out['cross_immunity'] = dict(rel_sev=ss.normal(loc=sev['par1'], scale=sev['par2']))
    if 'own_imm_hr' in v2_pars:
        out.setdefault('cross_immunity', {})['own_imm_hr'] = v2_pars['own_imm_hr']
    # v2 nested CD4 strata (lt200/gt200) became flat _lo/_hi on the HIV module.
    hiv2 = v2_pars.get('hiv_pars') or {}
    hiv3 = {}
    for effect in ('rel_sus', 'rel_sev', 'rel_imm'):
        if effect in hiv2:
            hiv3[f'{effect}_lo'] = hiv2[effect]['lt200']
            hiv3[f'{effect}_hi'] = hiv2[effect]['gt200']
    if 'rel_reactivation_prob' in hiv2:  # v3 split this by CD4 stratum
        hiv3['rel_reactivation_lo'] = hiv2['rel_reactivation_prob']
        hiv3['rel_reactivation_hi'] = hiv2['rel_reactivation_prob']
    if 'art_failure_prob' in hiv2:
        hiv3['p_effective_art'] = ss.bernoulli(p=1 - hiv2['art_failure_prob'])
    if hiv3:
        out['hiv'] = hiv3
    for gt, gpars in (v2_pars.get('genotype_pars') or {}).items():
        keep = {k: v for k, v in gpars.items() if k in ('cin_fn', 'cancer_fn')}
        if keep:
            out[gt] = {k: {kk: vv for kk, vv in v.items() if kk == 'k'}
                       for k, v in keep.items()}
    return out

""" 2. run_sim --> Simulation running function """

def run_sim(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_data=None,
        location='zambia', model_hiv=True):

    if hiv_data is None:
        hiv_data = _hiv_data()

    # Make sim
    sim = make_sim(
        debug=debug,
        seed=seed,
        end=end,
        hiv_data=hiv_data,
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
    """Return the top-n calibrated parameter sets with metadata.

    v3 removed Calibration.trial_pars_to_sim_pars. The parameter set for a
    trial is recovered from calib.df directly: every column other than the
    bookkeeping ones is a dotted parameter key, which route_pars re-expands
    into nested form when the sim is built.
    """
    df = calib.df.nsmallest(len(calib.df) if n is None else int(n), 'mismatch')
    bookkeeping = {'index', 'mismatch', 'rand_seed'}
    par_cols = [c for c in df.columns if c not in bookkeeping]
    top_pars = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        top_pars.append(dict(
            pars={c: row[c] for c in par_cols},
            trial_index=int(row['index']),
            mismatch=float(row['mismatch']),
            rank=rank,
        ))
    return top_pars

    
""" 3. run_multi_sim --> Multi-simulation function for running multiple simulations  """

def run_multi_sim(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_data=None,
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
    
    if hiv_data is None:
        hiv_data = _hiv_data()

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
                hiv_data=hiv_data,
                analyzers=analyzers,
                interventions=interventions,
                calib_pars=cfg_pars,
                model_hiv=model_hiv
            )
            base_sim['verbose'] = sim_verbose  # Use reduced verbosity for individual sims
            
            # Create and run MultiSim for this batch
            msim = ss.MultiSim(base_sim)
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
                    hiv_data=hiv_data,
                    analyzers=analyzers,
                    interventions=interventions,
                    calib_pars=cfg_pars,
                    model_hiv=model_hiv
                )
                final_msim = ss.MultiSim(final_base)
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


def _scale_art_coverage(hiv_data, scale):
    """Return hiv_data with ART coverage multiplied by `scale`, clipped to [0, 1].

    Used for the counterfactual-without-ART scenario (scale=0). v2 did this by
    writing scaled copies of the CSVs next to the originals; scaling the
    in-memory frame avoids littering data/ with derived files.
    """
    if scale == 1.0:
        return hiv_data
    out = dict(hiv_data)
    art = out['art_coverage'].copy()
    art['coverage'] = (art['coverage'] * scale).clip(0.0, 1.0)
    out['art_coverage'] = art
    return out


def run_multi_sim_optimized_art(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_data=None,
        n_runs=10, batch_size=None, top_pars=None, create_reduced=True,
        art_coverage_scale=1.0, model_hiv=True):
    """Wrapper around run_multi_sim with optional ART coverage scaling and HIV toggling."""
    if hiv_data is None:
        hiv_data = _hiv_data()

    return run_multi_sim(
        analyzers=analyzers,
        interventions=interventions,
        debug=debug,
        seed=seed,
        verbose=verbose,
        do_save=do_save,
        end=end,
        calib_pars=calib_pars,
        hiv_data=_scale_art_coverage(hiv_data, art_coverage_scale),
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


# v3 result names. The HIV-stratified keys and the age-standardized rate all
# live on the all_hpv analyzer (sim.results.all_hpv), not on sim.results
# directly as in v2; the HIV-stratified ones only exist when HIV is modelled.
DEFAULT_EXPORT_METRICS = [
    'new_cancers', 'cancers_with_hiv', 'cancers_no_hiv',
    'asr_cancer_incidence', 'cancer_incidence_with_hiv', 'cancer_incidence_no_hiv',
    'cancer_rate_ratio',
]


def _years(sim):
    """Float calendar years for a sim's result timeseries."""
    return np.asarray(sim.results.timevec.years, dtype=float)


def _result_series(sim, metric):
    """Return a sim's timeseries for `metric`, or None if it isn't present.

    v2 exposed every result flat on sim.results; v3 splits them across module
    result sets, with the pooled HPV and HIV-stratified cancer outputs on the
    all_hpv analyzer.
    """
    for holder in (sim.results.get('all_hpv'), sim.results):
        if holder is not None and metric in holder:
            return np.asarray(holder[metric], dtype=float)
    return None

""" 5. aggregate_metric_series --> Function to aggregate metric series across simulations """
def aggregate_metric_series(sims, metric, label_fmt='run_{rank:02d}'):
    """Return a DataFrame aggregating a single metric across sims."""
    columns = {'year': _years(sims[0])}

    collected = []
    for i, sim in enumerate(sims):
        series = _result_series(sim, metric)
        if series is None:
            continue
        rank = getattr(sim, 'rank', None)
        label = label_fmt.format(rank=rank if rank is not None else i)
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
        os.makedirs('results', exist_ok=True)
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
        years = _years(sim)
        sim_meta = dict(
            sim_index=sim_idx,
            rank=getattr(sim, 'rank', None),
            top_label=getattr(sim, 'top_label', getattr(sim, 'label', f'sim_{sim_idx}')),
            mismatch=getattr(sim, 'mismatch', None),
        )
        for metric in metrics:
            values = _result_series(sim, metric)
            if values is None:
                continue
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
    os.makedirs('results', exist_ok=True)
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

def create_age_analyzer(year=2020):
    """Create the age-stratified, HIV-stratified cancer analyzer."""
    return CancerByAgeHIV(years=year)


def aggregate_analyzer_results(sims, year=2020):
    """Aggregate analyzer results across simulations into DataFrames with statistics."""
    print('Aggregating analyzer results across all simulations...')

    metrics = ['cancers', 'cancers_with_hiv', 'cancers_no_hiv',
               'cancer_incidence_with_hiv', 'cancer_incidence_no_hiv',
               'cancer_rate_ratio']
    frames = []
    for i, sim in enumerate(sims):
        if i % 10 == 0:
            print(f'Processing simulation {i + 1}/{len(sims)}')
        analyzer = next((a for a in sim.analyzers.values()
                         if isinstance(a, CancerByAgeHIV)), None)
        if analyzer is None:
            print(f'Warning: no CancerByAgeHIV analyzer for simulation {i + 1}, skipping...')
            continue
        df = analyzer.to_dataframe(year)
        df['simulation_id'] = i + 1
        df['rank'] = getattr(sim, 'rank', np.nan)
        df['mismatch'] = getattr(sim, 'mismatch', np.nan)
        frames.append(df)

    if not frames:
        raise ValueError('No analyzer results found in any simulations')

    combined_df = pd.concat(frames, ignore_index=True)
    aggregate_stats = combined_df.groupby('bins', sort=False).agg(
        {m: ['median', q25_func, q75_func, 'mean', 'min', 'max'] for m in metrics}
    ).round(2)
    aggregate_stats.columns = ['_'.join(col).strip() for col in aggregate_stats.columns]
    aggregate_stats = aggregate_stats.reset_index()

    return combined_df, aggregate_stats


""" 8. run_multi_sim_with_analyzers --> Function to run multiple simulations with analyzers """

def run_multi_sim_with_analyzers(
        analyzers=None, interventions=None, debug=0, seed=1, verbose=0.5,
        do_save=False, end=2020, calib_pars=None, hiv_data=None,
        n_runs=100, batch_size=None, top_pars=None, create_reduced=True, model_hiv=True):
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
            hiv_data=hiv_data,
            n_runs=n_runs,
            batch_size=batch_size,
            top_pars=top_pars,
            create_reduced=create_reduced,
            model_hiv=model_hiv
        )

    # Otherwise, run with single parameter set (original behavior)
    if hiv_data is None:
        hiv_data = _hiv_data()

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
            hiv_data=hiv_data,
            analyzers=analyzers,
            interventions=interventions,
            calib_pars=calib_pars,
            model_hiv=model_hiv
        )
        base_sim['verbose'] = verbose

        # Create and run MultiSim for this batch
        msim = ss.MultiSim(base_sim)
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
                       do_save=False, end=2020, hiv_data=None,
                       n_runs=100, batch_size=25, location='zambia', output_dir='results'):
    """Run simulations for a single parameter set. Used for SLURM job array parallelization."""
    rank = par_set.get('rank')
    if rank is None:
        raise ValueError("par_set must include an integer 'rank'")
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
        hiv_data=hiv_data,
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
    
    return sims
