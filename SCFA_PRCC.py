import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager

# ------------------------- Experiment settings -------------------------
N_SAMPLES = 2000
N_BOOTSTRAP = 400
RELATIVE_RANGE = 0.20                 # baseline ±20%
SEED = 20261115
T_END_DAYS = 30.0 * 365.25
OUT_NAME = 'Figure_PRCC_G_beta_30yr'

PARAM_NAMES = (
    'Gmeal', 'PS', 'alphaS', 'etaS', 'gammaS',
    'rhoS', 'SI0', 'di', 'Ghep', 'kS', 'cG'
)

PARAM_LABELS = (
    r'$G_{\mathrm{meal}}$',
    r'$P_S$',
    r'$\alpha_S$',
    r'$\eta_S$',
    r'$\gamma_S$',
    r'$\rho_S$',
    r'$S_{I,0}$',
    r'$d_i$',
    r'$G_{\mathrm{hep}}$',
    r'$k_S$',
    r'$c_G$'
)
OUTPUT_LABELS = ('', '')

# ---------------------------- Model constants ---------------------------
SG, GREN, GHEP = 8.0, 107.0, 535.0
ALPHAS, KI, SI0, ETAS, KS = 0.0043, 0.05, 0.635, 7.2, 100.0
AG, SIGMA_G, GAMMA_S, KGLP, DI = 20000.0, 86400.0, 0.1, 100.0, 200.0
KB, SK, CG, RHO = 100.0, 4.0, 0.0037, 0.047
D0, A, GC, W = 0.0402, 0.0021, 103.0, 17.5
GMEAL0, PS0, G0 = 1310.0, 2330.0, 90.0

# Parameter order corresponds to PARAM_NAMES.
BASELINE = np.array([GMEAL0, PS0, ALPHAS, ETAS, GAMMA_S,
                     RHO, SI0, DI, GHEP, SK, CG], dtype=float)


def initial_state():
    """Original quasi-steady initial conditions (fixed over all samples)."""
    s = PS0 / (SK * (1.0 + CG * G0))
    si = SI0 * (1.0 + ETAS * s / (KS + s))
    c = si * G0
    y = GMEAL0 + GREN - SG * G0
    h = GHEP / (1.0 + ALPHAS * s)
    z = c - KI * y
    ins = 2.0 * (y + h) / (z + np.sqrt(z*z + 4.0*c*KI*(y+h)))
    sec = SIGMA_G * G0**2 / (AG + G0**2) * (1.0 + GAMMA_S*s/(KGLP+s))
    return np.array([G0, ins, DI*ins/sec, s], dtype=float)


def rhs(t, x, p):
    """Four-dimensional SCFA–T2D ODE from the original fitting code."""
    g, i, beta, s = x
    meal, ps, alpha_s, eta_s, gamma_s, rho_s, si0, di, ghep, ks, cg = p
    si = si0 * (1.0 + eta_s * s / (KS+s))
    sec = SIGMA_G*g*g/(AG+g*g) * (1.0+gamma_s*s/(KGLP+s))
    beta_growth = A/(1.0 + ((g-GC)/W)**4) - D0 + rho_s*s/(KB+s)
    return [meal+GREN+ghep/((1.0+alpha_s*s)*(1.0+KI*i)) - SG*g-si*i*g,
            sec*beta-di*i,
            beta_growth*beta,
            ps/(1.0+cg*g)-ks*s]


def simulate(p, y0):
    """Return [G(30 yr), beta(30 yr)] or None on numerical failure."""
    try:
        sol = solve_ivp(rhs, (0.0, T_END_DAYS), y0, args=(p,),
                        method='LSODA', t_eval=[T_END_DAYS],
                        rtol=2e-6, atol=[1e-6, 1e-9, 1e-13, 1e-6],
                        max_step=120.0)
        if (not sol.success or not np.isfinite(sol.y).all()
                or sol.y.min() < -1e-8 or sol.y[0].max() > 20000):
            return None
        return sol.y[[0, 2], -1]
    except (ValueError, RuntimeError, FloatingPointError, OverflowError):
        return None


def bootstrap_prcc(X, Y):
    """Bootstrap the partial rank correlations for the two outputs.

    X: (n, 11), Y: (n, 2); result: (2, n_bootstrap, 11).
    """
    n, k = X.shape
    out = np.full((2, N_BOOTSTRAP, k), np.nan)
    rng = np.random.default_rng(SEED+99)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        xr = rankdata(X[idx], axis=0)
        yr = rankdata(Y[idx], axis=0)
        xr = (xr-xr.mean(0)) / np.sqrt(((xr-xr.mean(0))**2).sum(0))
        yr = (yr-yr.mean(0)) / np.sqrt(((yr-yr.mean(0))**2).sum(0))
        cxx, cxy = xr.T@xr, xr.T@yr
        try:
            invxx = np.linalg.inv(cxx)
        except np.linalg.LinAlgError:
            continue
        a = invxx@cxy
        remaining_variance = np.maximum(1e-14, 1.0-np.sum(cxy*a, axis=0))
        denom = np.sqrt(np.diag(invxx)[:, None]*remaining_variance[None, :] + a*a)
        out[:, b, :] = np.clip(a/denom, -1, 1).T
    return out


def plot_prcc(bootstrap, output_dir):
    """Original two-panel journal style; labels below; tight horizontal gap."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((s for s in ('Times New Roman', 'Times', 'Tinos', 'Liberation Serif')
                 if s in installed), 'serif')
    plt.rcParams.update({'font.family': font, 'font.size': 17,
                         'axes.labelsize': 16, 'xtick.labelsize': 13,
                         'ytick.labelsize': 11, 'mathtext.fontset': 'dejavuserif',
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': 0.75, 'figure.facecolor': 'white',
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.15), sharey=True)
    for oi, ax in enumerate(axes):
        samples = [bootstrap[oi, :, j][np.isfinite(bootstrap[oi, :, j])]
                   for j in range(len(PARAM_NAMES))]
        box = ax.boxplot(samples, widths=.47, whis=1.5, patch_artist=True,
                         showfliers=False,
                         boxprops=dict(edgecolor='#5478d1', linewidth=.9),
                         medianprops=dict(color='#d94343', linewidth=1.1),
                         whiskerprops=dict(color='#6f7782', linewidth=.75),
                         capprops=dict(color='#6f7782', linewidth=.75))
        for patch in box['boxes']:
            patch.set_facecolor('#f7faff')
        for j, arr in enumerate(samples, 1):
            q1, q3 = np.percentile(arr, [25, 75]); delta = q3-q1
            outliers = arr[(arr < q1-1.5*delta) | (arr > q3+1.5*delta)]
            if len(outliers):
                inds = np.linspace(0, len(outliers)-1, min(5, len(outliers))).astype(int)
                ax.plot(np.full(len(inds), j), outliers[inds], '+', ms=3,
                        mec='#c64b4b', mew=.65, ls='none', color='#c64b4b')
        ax.axhline(0, ls=(0, (3, 3)), color='#444', lw=.8, zorder=0)
        ax.set_xlim(.4, len(PARAM_NAMES)+.6)
        ax.set_ylim(-1.02, 1.02)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.grid(axis='y', lw=.45, color='#ececec', zorder=0)
        ax.set_xticks(np.arange(1, len(PARAM_NAMES)+1), PARAM_LABELS)
        ax.tick_params(axis='x', length=0, pad=8)
        ax.tick_params(axis='y', direction='in', length=3)
        ax.set_ylabel('PRCC')
        ax.text(.5, -.19, f'({chr(ord("a")+oi)}) {OUTPUT_LABELS[oi]}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=13, clip_on=False)
        for spine in ax.spines.values():
            spine.set_color('#666')
    fig.subplots_adjust(left=.070, right=.985, bottom=.205, top=.975, wspace=.11)
    for ext in ('pdf', 'eps'):
        filename = output_dir / f'{OUT_NAME}.{ext}'
        fig.savefig(filename, bbox_inches='tight')
        print('Saved:', filename, flush=True)
    plt.close(fig)


def main():
    y0 = initial_state()
    lhs = qmc.LatinHypercube(d=len(BASELINE), seed=SEED).random(n=N_SAMPLES)
    Xraw = BASELINE * (1.0 + RELATIVE_RANGE*(2.0*lhs-1.0))
    valid_params, valid_outputs = [], []
    for i, p in enumerate(Xraw, 1):
        result = simulate(p, y0)
        if result is not None:
            valid_params.append(p)
            valid_outputs.append(result)
        if i % 500 == 0:
            print(f'Simulated {i}/{N_SAMPLES}; valid: {len(valid_outputs)}', flush=True)
    X, Y = np.asarray(valid_params), np.asarray(valid_outputs)
    if len(X) < max(100, len(BASELINE)+12):
        raise RuntimeError('Too few valid ODE solutions for PRCC.')
    print('Computing bootstrap PRCC...', flush=True)
    bootstrap = bootstrap_prcc(X, Y)
    plot_prcc(bootstrap, Path(__file__).resolve().parent)
    print(f'Finished: {len(X)}/{N_SAMPLES} valid samples at t=30 years.', flush=True)


if __name__ == '__main__':
    main()
