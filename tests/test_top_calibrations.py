"""Regression tests for run_functions helpers used by run_top_calibrations.py.

Covers three bugs found in the engineering-uplift review: get_top_calibrated_pars
crashing on its own default (n=None), aggregate_metric_series crashing on sims
without an explicit rank, and run_single_par_set silently accepting a missing rank.
"""
import numpy as np
import pandas as pd
import sciris as sc
import pytest
import run_functions as rf


class FakeCalib:
    """Stands in for a v3 hpv.Calibration: the par set is read off df directly,
    since v3 removed trial_pars_to_sim_pars."""
    df = pd.DataFrame({'index': [0, 1, 2], 'mismatch': [0.1, 0.2, 0.3],
                       'beta': [0.1, 0.2, 0.3]})


class FakeTimevec:
    years = np.array([2020.0, 2021.0])


class FakeSim:
    """Mimics the v3 result layout: a timevec with .years, and the pooled HPV
    results on an all_hpv sub-object rather than flat on sim.results."""

    def __init__(self, label, rank=None):
        self.label = label
        self.rank = rank
        self.results = sc.objdict(timevec=FakeTimevec(),
                                  all_hpv=sc.objdict(new_cancers=np.array([1.0, 2.0])))


def test_get_top_calibrated_pars_default_n():
    top = rf.get_top_calibrated_pars(FakeCalib())
    assert len(top) == 3


def test_get_top_calibrated_pars_n_limits_results():
    top = rf.get_top_calibrated_pars(FakeCalib(), n=2)
    assert len(top) == 2


def test_aggregate_metric_series_without_rank():
    sims = [FakeSim(f'Sim--{i}') for i in range(3)]
    df = rf.aggregate_metric_series(sims, 'new_cancers')
    assert list(df['year']) == [2020, 2021]
    assert 'mean' in df.columns


def test_aggregate_metric_series_skips_absent_metric():
    """A metric missing from the sim (e.g. HIV-stratified keys with HIV off)
    yields None rather than raising."""
    assert rf.aggregate_metric_series([FakeSim('s')], 'cancers_with_hiv') is None


def test_run_single_par_set_requires_rank():
    with pytest.raises(ValueError):
        rf.run_single_par_set({'pars': {}}, n_runs=1)
