"""Smoke test for the Zambia baseline sim: an uncalibrated debug-mode sim builds and runs."""
import run_functions as rf


def test_run_sim_debug_runs():
    sim = rf.run_sim(debug=1, end=2020)
    assert sim.results['year'][-1] == 2020


def test_analyzers_are_passed_through():
    import hpvsim as hpv
    az = hpv.age_pyramid(timepoints=['2020'])
    hiv_datafile, art_datafile = rf._default_datafiles('zambia')
    sim = rf.make_sim(debug=1, end=2020, analyzers=[az],
                       hiv_datafile=hiv_datafile, art_datafile=art_datafile)
    assert len(sim['analyzers']) == 1
