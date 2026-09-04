"""Reproduce the manuscript figures from the CSVs run_counterfactuals.py writes.

Runs locally in seconds and needs no simulation: everything comes from
results/zambia_figure_{timeseries,by_age}.csv, plus results/zambia_calib.obj
for the supplementary posterior densities.

    python run_counterfactuals.py --run-sim     # on the VM, once
    python plot_figures.py                      # locally, as often as you like

Figures are written to figures/ (gitignored -- they are regenerable output).

Colours: the manuscript's Figure 2 used blue/green/red for the three
scenarios, which fails a colourblind-safety check -- the green and red sit at
deutan Delta-E 5.3, so the No-ART and Status-Quo lines, whose divergence is the
paper's ART result, are indistinguishable to roughly 8% of male readers. The
Okabe-Ito set below passes at 11.4. Figure 3's original orange/teal passes as
published and is kept.
"""

import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt

LOCATION = 'zambia'
FIGDIR = 'figures'

# Rolling-window (years) applied to the time-series panels (Figure 2 and
# Figure 3a) before plotting: cancer counts per year are stochastic at
# modest ensemble sizes, and the manuscript's smooth curves came from much
# larger ensembles. Applied to the median, lo, and hi columns per group so
# the IQR band and the median move together.
SMOOTH_WINDOW = 3

# Fixed assignment, never cycled: scenario -> (colour, label). Matches the
# manuscript palette so the reproduced figure aligns visually with what John
# published. The published green/red pair fails a colourblind check at deutan
# delta-E 5.3, which is what the earlier Okabe-Ito palette was picked to fix;
# see git history if the colourblind concern re-surfaces.
SCENARIOS = {
    'no_hiv':     ('#4A90D9', 'Scenario 1: No HIV'),
    'no_art':     ('#7FB77E', 'Scenario 2: No ART'),
    'status_quo': ('#D9455F', 'Status Quo'),
}
HIV_STATUS = {
    'with_hiv': ('#D95F0E', 'Women with HIV'),
    'no_hiv':   ('#1B9E77', 'Women without HIV'),
}
TARGET_STYLE = dict(color='#CC3311', marker='s', linestyle='none', markersize=5,
                    zorder=5)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False,
})


def _load():
    ts = pd.read_csv(f'results/{LOCATION}_figure_timeseries.csv')
    by_age = pd.read_csv(f'results/{LOCATION}_figure_by_age.csv')
    return ts, by_age


def _bin_order(bins):
    """Age-bin labels sorted by their lower bound, not lexically."""
    return sorted(set(bins), key=lambda b: float(str(b).split('-')[0]))


def _quantiles(df, index):
    """Median and interquartile range across parameter sets."""
    g = df.groupby(index)['value']
    return pd.DataFrame({'median': g.median(), 'lo': g.quantile(0.25),
                         'hi': g.quantile(0.75)}).reset_index()


def _smooth(s, window=SMOOTH_WINDOW, cols=('median', 'lo', 'hi')):
    """Centered rolling mean over `window` years; edges use whatever data is
    available so the curve reaches both ends of the plotted range."""
    out = s.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].rolling(window=window, center=True, min_periods=1).mean()
    return out


def _targets(fname, name):
    """Long-format target CSV -> {age lower bound: value}, females only."""
    df = pd.read_csv(f'data/{LOCATION}_{fname}.csv')
    df = df[df['name'] == name]
    if 'sex' in df.columns:
        df = df[df['sex'].astype(str).str.lower().str.startswith('f')]
    if 'age' not in df.columns:
        return float(df['value'].iloc[0])
    return dict(zip(df['age'].astype(float), df['value'].astype(float)))


def _boxplot_by_bin(ax, df, bins, ylabel, title):
    """Box per age bin across parameter sets; whiskers to the extremes, as published."""
    data = [df.loc[df['bins'] == b, 'value'].dropna().values for b in bins]
    ax.boxplot(data, whis=(0, 100), widths=0.6, patch_artist=True,
               boxprops=dict(facecolor='#AEC7E8', edgecolor='#33393F', linewidth=0.8),
               medianprops=dict(color='#33393F', linewidth=1.4),
               whiskerprops=dict(color='#33393F', linewidth=0.8),
               capprops=dict(color='#33393F', linewidth=0.8), showfliers=False)
    ax.set_xticks(range(1, len(bins) + 1))
    ax.set_xticklabels([str(b).split('-')[0] for b in bins], rotation=90)
    ax.set_xlabel('Age')
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def figure1(ts, by_age, year=2020):
    """Calibration: cases by age, ASIR, and cancer IRR by HIV status."""
    sq_age = by_age[(by_age['scenario'] == 'status_quo') & (by_age['year'] == year)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) cases by age vs GLOBOCAN
    cases = sq_age[sq_age['metric'] == 'cancers']
    bins = _bin_order(cases['bins'])
    _boxplot_by_bin(axes[0, 0], cases, bins, 'Number of cases',
                    f'Cancer cases by age, {year}')
    tgt = _targets('cancer_cases', 'cancers')
    xs = [i + 1 for i, b in enumerate(bins) if float(str(b).split('-')[0]) in tgt]
    ys = [tgt[float(str(b).split('-')[0])] for b in bins
          if float(str(b).split('-')[0]) in tgt]
    axes[0, 0].plot(xs, ys, label='GLOBOCAN estimates', **TARGET_STYLE)
    axes[0, 0].legend(loc='upper right')

    # (b) age-standardised incidence rate
    asir = ts[(ts['scenario'] == 'status_quo') & (ts['year'] == year)
              & (ts['metric'] == 'asr_cancer_incidence')]['value'].dropna()
    ax = axes[0, 1]
    ax.boxplot([asir.values], whis=(0, 100), widths=0.4, patch_artist=True,
               boxprops=dict(facecolor='#AEC7E8', edgecolor='#33393F', linewidth=0.8),
               medianprops=dict(color='#33393F', linewidth=1.4), showfliers=False)
    ax.plot([1], [_targets('asr_cancer_incidence', 'asr_cancer_incidence')],
            label='GLOBOCAN estimate', **TARGET_STYLE)
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    ax.set_ylabel('ASIR per 100,000')
    ax.set_title(f'Age-standardized incidence rate, {year}')
    # Lower left, so the legend swatch doesn't sit beside the plotted target
    # and read as a second data point.
    ax.legend(loc='lower left')

    # (c) cancer incidence rate ratio by age. Restricted to 25-75, matching
    # the manuscript: below 25 almost no cancers occur in either stratum, and
    # above 75 the WWH denominator is a few hundred women, so a handful of
    # events produce whiskers that dominate the y-axis and hide the signal.
    irr = sq_age[sq_age['metric'] == 'cancer_rate_ratio']
    bins_irr = [b for b in _bin_order(irr['bins'])
                if 25 <= float(str(b).split('-')[0]) <= 75]
    irr = irr[irr['bins'].isin(bins_irr)]
    _boxplot_by_bin(axes[1, 0], irr, bins_irr, 'IRR',
                    f'Cancer IRR: women with vs without HIV, {year}')
    tgt = _targets('cancer_rate_ratios', 'cancer_hiv_rate_ratios')
    xs = [i + 1 for i, b in enumerate(bins_irr) if float(str(b).split('-')[0]) in tgt]
    ys = [tgt[float(str(b).split('-')[0])] for b in bins_irr
          if float(str(b).split('-')[0]) in tgt]
    axes[1, 0].plot(xs, ys, label='Cancer registry estimates', **TARGET_STYLE)
    axes[1, 0].legend(loc='upper right')

    axes[1, 1].axis('off')
    for ax, tag in zip(axes.flat, 'abc'):
        ax.set_title(ax.get_title(), loc='center')
        ax.text(-0.12, 1.06, f'({tag})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom')
    fig.tight_layout()
    return fig


def figure2(ts, start=1990, stop=2025):
    """Crude cancer incidence under the three scenarios, median and IQR."""
    df = ts[(ts['metric'] == 'cancer_incidence') & ts['year'].between(start, stop)]
    q = _quantiles(df, ['scenario', 'year'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (colour, label) in SCENARIOS.items():
        s = q[q['scenario'] == name].sort_values('year')
        if s.empty:
            continue
        s = _smooth(s)
        ax.fill_between(s['year'], s['lo'], s['hi'], color=colour, alpha=0.2, linewidth=0)
        ax.plot(s['year'], s['median'], color=colour, linewidth=2, label=label)
    ax.set_xlabel('Year')
    ax.set_ylabel('Cancer incidence rate\n(per 100,000 women)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def figure3(ts, by_age, start=1990, stop=2025, year=2025):
    """Cancer incidence by HIV status: over time, and age-specific."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8))

    # (a) over time, status quo
    df = ts[(ts['scenario'] == 'status_quo') & ts['year'].between(start, stop)]
    for key, (colour, label) in HIV_STATUS.items():
        s = _quantiles(df[df['metric'] == f'cancer_incidence_{key}'], ['year'])
        if s.empty:
            continue
        s = _smooth(s.sort_values('year'))
        axes[0].fill_between(s['year'], s['lo'], s['hi'], color=colour,
                             alpha=0.2, linewidth=0)
        axes[0].plot(s['year'], s['median'], color=colour, linewidth=2, label=label)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Cervical cancer incidence rate\n(per 100,000 women)')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='upper left')

    # (b) age-specific, given year
    sq = by_age[(by_age['scenario'] == 'status_quo') & (by_age['year'] == year)]
    bins = _bin_order(sq['bins'])
    x = np.arange(len(bins))
    width = 0.38
    for i, (key, (colour, label)) in enumerate(HIV_STATUS.items()):
        s = _quantiles(sq[sq['metric'] == f'cancer_incidence_{key}'], ['bins'])
        s = s.set_index('bins').reindex(bins)
        axes[1].bar(x + (i - 0.5) * width, s['median'], width * 0.94, color=colour,
                    label=label, yerr=[s['median'] - s['lo'], s['hi'] - s['median']],
                    error_kw=dict(ecolor='#6C737A', elinewidth=0.8, capsize=2))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bins, rotation=90)
    axes[1].set_xlabel('Age group')
    axes[1].set_ylabel(f'Age-specific cancer incidence rate ({year})\n'
                       '(median with IQR)')
    axes[1].legend(loc='upper right')

    for ax, tag in zip(axes, 'ab'):
        ax.text(-0.10, 1.04, f'({tag})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom')
    fig.tight_layout()
    return fig


def supplementary_figure1():
    """Posterior density of each calibrated parameter against its uniform prior."""
    calib = sc.loadobj(f'results/{LOCATION}_calib.obj')
    df, spec = calib.df, calib.calib_pars
    pars = [c for c in df.columns if c not in ('index', 'mismatch', 'rand_seed')]
    ncol = 4
    nrow = int(np.ceil(len(pars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow))
    for ax, par in zip(axes.flat, pars):
        vals = df[par].astype(float).dropna().values
        lo, hi = spec[par]['low'], spec[par]['high']
        # Gaussian KDE needs spread; a parameter Optuna pinned gets a rug instead.
        if len(np.unique(vals)) > 2:
            from scipy.stats import gaussian_kde
            grid = np.linspace(lo, hi, 200)
            ax.fill_between(grid, gaussian_kde(vals)(grid), color='#AEC7E8',
                            edgecolor='#5B8FBF', linewidth=0.8)
        else:
            ax.plot(vals, np.zeros_like(vals), '|', color='#5B8FBF', markersize=12)
        ax.axhline(1.0 / (hi - lo), color='#8C2D19', linestyle='--', linewidth=1.2)
        ax.set_title(par, fontsize=8)
        ax.set_xlim(lo, hi)
    for ax in axes.flat[len(pars):]:
        ax.axis('off')
    fig.supxlabel('Value')
    fig.supylabel('Density')
    fig.suptitle('Posterior (shaded) against uniform prior (dashed), '
                 f'top {len(df)} parameter sets', fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs(FIGDIR, exist_ok=True)
    ts, by_age = _load()
    for name, fig in (('figure1_calibration', figure1(ts, by_age)),
                      ('figure2_scenarios', figure2(ts)),
                      ('figure3_by_hiv_status', figure3(ts, by_age)),
                      ('supplementary_figure1_posteriors', supplementary_figure1())):
        path = f'{FIGDIR}/{name}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {path}')
