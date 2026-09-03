"""Tests for the v2.2.6 -> v3 migration helpers and the Zambia-local analyzer."""
import json

import numpy as np
import starsim as ss
import pytest

import run_functions as rf
from analyzers import CancerByAgeHIV


def test_v2_pars_translate_to_v3():
    """The saved v2 parameter set maps onto names v3 actually accepts."""
    v2 = json.load(open('results/v2_artefact_snapshot.json'))['pars_nov06']
    out = rf.v2_calib_pars_to_v3(v2, dt=0.25)

    # Cross-layer probabilities are per-timestep in v2.2.6, annual in v3.
    assert out['m_cross_layer'] == pytest.approx(1 - (1 - 0.35) ** 4)
    assert out['f_cross_layer'] == pytest.approx(1 - (1 - 0.1) ** 4)

    # Partner counts are Poisson rates, not probabilities: no conversion, and
    # v2's poisson1 offset is now the network's job.
    assert out['m_partners_casual'].pars.lam == pytest.approx(0.34)
    assert out['f_partners_marital'].pars.lam == pytest.approx(0.01)

    # v2 sev_dist is v3's CrossImmunity.rel_sev; v2 CD4 strata flatten to _lo/_hi.
    assert out['cross_immunity']['rel_sev'].pars.loc == pytest.approx(1.33)
    assert out['hiv']['rel_sev_lo'] == 1.5 and out['hiv']['rel_sev_hi'] == 1.2
    assert out['hiv']['rel_reactivation_lo'] == out['hiv']['rel_reactivation_hi'] == 3
    # art_failure_prob has no hpvsim par in v3; stisim's p_effective_art is its complement.
    assert out['hiv']['p_effective_art'].pars.p == pytest.approx(0.9)

    # Only the genotype pars v3 still has are carried over.
    assert out['hpv16'] == {'cin_fn': {'k': 0.35}, 'cancer_fn': {'k': 0.25}}


def test_translated_v2_pars_build_a_sim():
    """The translated names survive route_pars, which is strict about unknowns."""
    v2 = json.load(open('results/v2_artefact_snapshot.json'))['pars_nov06']
    sim = rf.make_sim(debug=1, stop=1985, calib_pars=rf.v2_calib_pars_to_v3(v2, dt=1.0))
    sim.init()
    net = sim.networks.sexualnetwork
    assert net.pars.m_cross_layer == pytest.approx(0.35)  # dt=1 -> identity
    assert sim.diseases.hiv.pars.rel_reactivation_lo == 3


def test_hiv_counts_are_scale_correct():
    """hiv_counts weights by per-agent scale; sim.results.hiv.n_infected does not.

    At ms_agent_ratio > 1 the stisim stock counts fine agents at full weight and
    then applies pop_scale, over-reporting HIV. Prevalence must stay a fraction.
    """
    sim = rf.make_sim(debug=1, stop=1990, seed=1)
    sim.run()
    n_hiv, prev = rf.hiv_counts(sim)
    assert 0 < prev < 1
    assert n_hiv > 0
    # 15-49 is a subset, so its headcount cannot exceed the all-age one.
    n_adult, _ = rf.hiv_counts(sim, 15, 50)
    assert n_adult <= n_hiv
    # The unweighted result is the biased one, and is strictly larger here.
    assert float(sim.results.hiv.n_infected[-1]) > n_hiv


def test_cancer_by_age_hiv_analyzer():
    """The Layer-3 analyzer reports age-binned counts split by HIV status."""
    az = CancerByAgeHIV(years=1985)
    sim = rf.make_sim(debug=1, stop=1985, analyzers=[az])
    sim.run()
    analyzer = next(a for a in sim.analyzers.values() if isinstance(a, CancerByAgeHIV))
    df = analyzer.to_dataframe(1985)

    assert len(df) == len(analyzer.bin_labels)
    assert {'cancers', 'cancers_with_hiv', 'cancers_no_hiv',
            'cancer_incidence_with_hiv', 'cancer_rate_ratio'} <= set(df.columns)
    # Total is the sum of the two strata, and counts are non-negative.
    assert np.allclose(df['cancers'], df['cancers_with_hiv'] + df['cancers_no_hiv'])
    assert (df[['cancers_with_hiv', 'cancers_no_hiv']] >= 0).all().all()
    # Cancer is female-only, so the under-15 bins should be empty.
    assert df.loc[df['bins'].isin(['0-5', '5-10']), 'cancers'].sum() == 0

    with pytest.raises(ValueError, match='not in the sim window'):
        analyzer.to_dataframe(2050)
