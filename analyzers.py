"""Zambia-specific analyzers.

hpvsim v3's `hpv.by_age` records age-binned results but has no HIV
stratification, and its key whitelist covers only `cancers` of the seven
outputs the Zambia analysis needs. `CancerByAgeHIV` below fills that gap
locally. It follows the probe pattern used in hpvsim's own
`tests/regression/plot_rwanda_calib.py`, which does the same job for Rwanda.

If a third HIV-HPV country needs this, it is a candidate for promotion into
hpvsim proper (as an `hiv_stratified=` option on `by_age`), which would also
make age-by-HIV targets reachable through `hpv.Calibration(data=)`.
"""

import numpy as np
import pandas as pd
import starsim as ss

import hpvsim as hpv
from hpvsim.hpv import HPV

__all__ = ['CancerByAgeHIV']

_DEFAULT_EDGES = np.array([0., 5., 10., 15., 20., 25., 30., 35., 40.,
                           45., 50., 55., 60., 65., 70., 75., 80., 100.])


class CancerByAgeHIV(ss.Analyzer):
    """Age-binned cervical cancer counts and incidence, stratified by HIV status.

    Accumulates scale-weighted new female cancers and female person-time per
    (age bin, HIV status, timestep). Incidence is derived per calendar year as
    cancers summed over the year divided by mean female headcount in that year,
    per 100k person-years.

    Args:
        edges: age bin edges. Default 5-year bins to 80, then 80-100.
        years: calendar years to report from `to_dataframe`. Default: all.

    Example::

        az = CancerByAgeHIV(years=2020)
        sim = make_sim(analyzers=[az]); sim.run()
        df = az.to_dataframe(2020)   # one row per age bin
    """

    def __init__(self, edges=None, years=None, **kwargs):
        super().__init__(**kwargs)
        self.edges = np.asarray(_DEFAULT_EDGES if edges is None else edges, dtype=float)
        self.years = None if years is None else sorted(int(y) for y in np.atleast_1d(years))
        self.bin_labels = [f'{int(lo)}-{int(hi)}' for lo, hi
                           in zip(self.edges[:-1], self.edges[1:])]
        self.hpv_modules = None
        self.hiv_module = None
        # Filled in init_pre: (n_bins, n_timesteps) per HIV status.
        self.cancers = None
        self.female_time = None
        self._year_of_ti = None

    def init_pre(self, sim):
        super().init_pre(sim)
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        self.hiv_module = hpv.misc.hiv_module(sim)
        n_bin, n_ti = len(self.bin_labels), len(sim.t.timevec)
        self.cancers = {s: np.zeros((n_bin, n_ti)) for s in ('pos', 'neg')}
        self.female_time = {s: np.zeros((n_bin, n_ti)) for s in ('pos', 'neg')}
        self._year_of_ti = np.floor(np.asarray(sim.t.timevec, dtype=float)).astype(int)

    def step(self):
        # sim.ti, not self.ti: the HIV module carries its own dt, and cancer
        # events must be counted on the sim's clock to avoid double-counting.
        ti = self.sim.ti
        people = self.sim.people
        alive = people.alive.values
        # Multiscale fine agents carry scale < 1, so weight rather than count.
        weight = people.scale.values
        female = people.female.values & alive
        age = people.age.values

        if self.hiv_module is None:
            pos = np.zeros(alive.shape, dtype=bool)
        else:
            pos = self.hiv_module.infected.values
        status = {'pos': female & pos, 'neg': female & ~pos}

        # Gate on the in-state flag as well as the event time: a scheduled
        # ti_cancerous survives on agents who died of a competing cause, so a
        # bare time match over-counts.
        new_cancer = np.zeros(alive.shape, dtype=bool)
        for module in self.hpv_modules:
            new_cancer |= (module.cancerous.values & (module.ti_cancerous.values == ti))

        for bi, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            in_bin = (age >= lo) & (age < hi)
            for key, mask in status.items():
                self.cancers[key][bi, ti] = (weight * (new_cancer & mask & in_bin)).sum()
                self.female_time[key][bi, ti] = (weight * (mask & in_bin)).sum()

    def to_dataframe(self, year):
        """Age-binned counts, incidence and rate ratio for one calendar year."""
        sel = self._year_of_ti == int(year)
        if not sel.any():
            raise ValueError(f'{type(self).__name__}: year {year} not in the sim window')
        scale = self.sim.pars.pop_scale
        out = {'bins': self.bin_labels}
        for key, suffix in (('pos', '_with_hiv'), ('neg', '_no_hiv')):
            cancers = self.cancers[key][:, sel].sum(axis=1) * scale
            person_years = self.female_time[key][:, sel].mean(axis=1) * scale
            out[f'cancers{suffix}'] = cancers
            with np.errstate(divide='ignore', invalid='ignore'):
                out[f'cancer_incidence{suffix}'] = np.where(
                    person_years > 0, cancers / person_years * 1e5, np.nan)
        out['cancers'] = out['cancers_with_hiv'] + out['cancers_no_hiv']
        # NaN rather than 0 where the HIV- rate is zero: a zero denominator
        # means "no information", and 0.0 would drag a mean or a fit downward.
        with np.errstate(divide='ignore', invalid='ignore'):
            out['cancer_rate_ratio'] = np.where(
                out['cancer_incidence_no_hiv'] > 0,
                out['cancer_incidence_with_hiv'] / out['cancer_incidence_no_hiv'], np.nan)
        return pd.DataFrame(out)
