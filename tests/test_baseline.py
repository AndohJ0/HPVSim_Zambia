"""Smoke test for the Zambia baseline sim: an uncalibrated debug-mode sim builds and runs."""
import numpy as np
import run_functions as rf


def test_run_sim_debug_runs():
    sim = rf.run_sim(debug=1, end=2020)
    assert sim.results.timevec.years[-1] == 2020
    # HIV is modelled by default, so the stratified cancer results should exist.
    assert 'cancers_with_hiv' in sim.results.all_hpv


def test_analyzers_are_passed_through():
    import hpvsim as hpv
    az = hpv.age_pyramid(timepoints=['2020'])
    sim = rf.make_sim(debug=1, stop=2020, analyzers=[az])
    assert any(isinstance(a, hpv.age_pyramid) for a in sim.pars.analyzers)


def test_hiv_data_matches_v3_contract():
    """_hiv_data supplies the keys and columns hpv.Sim(hiv_data=) requires."""
    data = rf._hiv_data()
    assert set(data) == {'incidence', 'art_coverage'}
    assert list(data['incidence'].columns) == ['age', 'sex', 'year', 'incidence']
    assert list(data['art_coverage'].columns) == ['age', 'sex', 'year', 'coverage']
    # The by-age ART CSVs carry a BOM and ~500 trailing blank rows.
    assert not data['art_coverage'].isna().any().any()
    assert set(data['art_coverage']['sex']) == {'f', 'm'}
    assert data['art_coverage']['coverage'].between(0, 1).all()


def test_layer_probs_annualized():
    """Per-timestep v2.2.6 probabilities are converted up, not down."""
    assert np.isclose(rf._to_annual_prob(0.1, 0.25), 0.3439, atol=1e-4)
    annual = rf._layer_probs_to_annual(rf._LAYER_PROBS_MARITAL, 0.25)
    # Row 0 holds age-bin bounds and must be untouched; rows 1-2 rise.
    assert np.array_equal(annual[0], rf._LAYER_PROBS_MARITAL[0])
    assert (annual[1:3] >= rf._LAYER_PROBS_MARITAL[1:3]).all()
    assert (annual[1:3] <= 1).all()
