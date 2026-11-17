import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
P = Path(__file__).resolve().parent
OUT = P / 'figures'
OBSERVATIONS_CSV = "subject,year,glucose\n1,11.3,89.0\n1,13.4,92.0\n1,15.6,86.9\n1,17.5,86.6\n1,22.0,90.5\n1,27.6,87.3\n1,30.8,93.6\n1,33.5,92.4\n1,36.3,104.1\n1,42.1,103.6\n1,44.1,120.7\n1,45.8,177.6\n2,8.1,85.0\n2,9.9,88.6\n2,12.1,82.8\n2,16.6,86.7\n2,19.0,89.1\n2,21.4,98.1\n2,27.0,94.6\n2,29.4,95.9\n2,33.1,95.0\n2,34.8,101.6\n2,39.9,136.1\n3,8.6,91.8\n3,14.2,88.0\n3,18.9,87.1\n3,24.5,82.8\n3,26.7,84.0\n3,28.8,84.9\n3,30.8,85.5\n3,31.9,91.2\n3,32.2,93.1\n3,35.7,95.2\n3,41.8,142.9\n4,5.8,92.4\n4,8.2,89.2\n4,10.9,81.0\n4,13.2,87.6\n4,17.0,87.5\n4,21.5,85.3\n4,23.5,82.1\n4,26.6,96.6\n4,29.6,115.9\n4,31.5,150.5\n4,37.9,92.7\n5,39.7,94.2\n5,41.7,107.9\n5,44.0,101.4\n5,46.1,89.4\n5,48.0,107.9\n5,49.8,102.6\n5,51.9,96.2\n5,54.1,101.8\n5,56.7,115.2\n5,58.6,105.7\n5,60.5,105.0\n5,66.4,115.3\n5,69.7,123.0\n6,21.7,88.6\n6,31.7,87.0\n6,36.1,99.9\n6,37.7,102.1\n6,38.2,102.8\n6,40.3,109.9\n6,43.0,110.9\n6,44.9,113.6\n6,46.9,116.7\n7,16.3,88.9\n7,26.3,94.4\n7,29.7,93.3\n7,34.0,84.7\n7,36.9,92.2\n7,43.6,94.6\n7,45.4,116.8\n7,47.5,152.9\n8,30.3,88.4\n8,36.1,87.4\n8,38.8,96.3\n8,41.0,96.6\n8,42.9,91.8\n8,45.1,97.5\n8,47.5,99.2\n8,49.5,99.6\n8,51.5,177.5\n9,7.5,92.2\n9,9.3,99.2\n9,11.8,90.7\n9,14.1,97.9\n9,16.9,92.5\n9,19.5,95.8\n9,22.7,86.7\n9,24.9,97.4\n9,26.6,99.0\n9,28.6,113.0\n9,31.8,196.0\n10,22.2,85.4\n10,26.0,86.4\n10,28.2,98.3\n10,30.2,97.9\n10,32.2,97.0\n10,35.4,106.0\n10,37.9,106.0\n10,40.8,112.7\n10,43.3,159.2\n11,23.0,87.1\n11,25.8,92.0\n11,27.0,86.3\n11,29.1,83.5\n11,31.5,90.6\n11,33.0,97.6\n11,37.3,94.8\n11,40.8,95.7\n11,42.9,90.3\n11,45.0,91.6\n11,49.1,95.5\n11,52.2,100.0\n11,59.5,102.5"
ISLAM_GMEAL_G0 = {1: (2322.0, 94.0), 2: (2324.0, 94.0), 3: (2325.0, 92.0), 4: (2306.54, 97.63), 5: (2316.05, 97.38), 6: (2420.97, 97.1), 7: (2215.84, 93.19), 8: (2177.9, 99.0), 9: (2197.59, 73.7), 10: (2270.26, 98.87), 11: (1700.0, 90.0)}
SMOOTH6_STARTS = 'subject,Gmeal,PS,G0,I0,beta0,S0\n1,1381.85073846983,2331.1278115664,87.0704165772,2.52119728566847,0.0200376134221957,441.477836884565\n2,1908.87589892039,2331.55317783242,83.238894167191,4.13005882095641,0.0343556487920105,445.604459010154\n3,1528.36913116498,2331.53454857675,89.8460089652605,2.72472946444205,0.0211324122604204,456.078098104545\n4,2067.40494933919,2329.71391864169,89.4048838180526,4.14152452888402,0.0310560853735341,437.707249087022\n5,1291.38748456034,2337.66075028113,96.6794559091295,1.89479454934774,0.0121251321505389,443.679483758212\n6,1285.00000009196,2327.56244473628,88.6656088818921,2.19953668009851,0.0166858924318641,438.149936762888\n7,1599.91268178612,2330.99884260954,88.4925348295916,3.00758379320523,0.022880330195155,439.042002305487\n8,2356.99883240707,2326.79802260559,87.6144546202333,5.00407095491226,0.0386177326296392,439.292519837461\n9,2130.15306974173,2329.22589739712,92.1683018031478,4.12520675335249,0.029615630793685,434.288200431491\n10,1285.55271519712,2322.04594788221,84.9253347201187,2.36751699617392,0.0193237338913484,443.532733265154\n11,1285.04766232507,2351.43012223459,86.4788856261311,2.29740636462658,0.0181281499216293,446.565300691892'

def smooth6_seed():
    return pd.read_csv(StringIO(SMOOTH6_STARTS)).set_index('subject')
REFERENCE_FITTED_CSV = 'subject,Gmeal,PS,G0,log_beta_multiplier,RMSE\n1,1370.1301775902984,2331.1311775265167,87.473899536119418,0,5.598671407619821\n2,1908.8831924543197,2331.5525015078224,83.238428102034419,0,5.6435432691905856\n3,1491.8799673905371,2331.5051562922899,89.830692596099226,0,9.8661787356577832\n4,2067.457039737822,2329.7144677107917,89.402418375870496,0,9.9323377019656967\n5,1300.0257991456251,2331.0528893793548,94.241878427800074,0,8.1774722844784478\n6,1285.000000091959,2327.5624447362834,88.665608881892155,0,5.6958221732315133\n7,1599.9048712445071,2330.9985840639961,88.492449247362458,0,7.1384705475876409\n8,2356.998832407065,2326.7980226055893,87.614454620233346,0,5.5256787773185643\n9,2130.1502814166724,2329.2220571964872,92.168645346963103,0,5.2174466270666082\n10,1285.0289040432588,2322.0741025812017,84.906926704674717,0,3.711481902007304\n11,1285.0000000000002,2351.428973184396,86.466907480052825,0,4.8985713754692553'
D = 365.25
SG = 8.0
GREN = 107.0
GHEP = 535.0
ALPHAS = 0.0043
KI = 0.05
SI0 = 0.635
ETAS = 7.2
KS = 100.0
AG = 20000.0
SIGMA_G = 86400.0
GAMMA_S = 0.1
KGLP = 100.0
DI = 200.0
KB = 100.0
SK = 4.0
CG = 0.0037
RHO = 0.047
SH = (0.0402, 0.0021, 103.0, 17.5)
QSS_LOG_PENALTY = 1.0

def load_observations():
    observations = pd.read_csv(StringIO(OBSERVATIONS_CSV))
    assert len(observations) == 117 and observations.subject.nunique() == 11
    observations = observations.sort_values(['subject', 'year']).reset_index(drop=True)
    print(f'内置数据：{len(observations)} 个血糖点，{observations.subject.nunique()} 名受试者')
    return observations

def sensitivity(s):
    return SI0 * (1.0 + ETAS * s / (KS + s))

def secretion(g, s):
    return SIGMA_G * g * g / (AG + g * g) * (1.0 + GAMMA_S * s / (KGLP + s))

def growth(g, s, sh=SH):
    d0, A, Gc, w = sh
    return A / (1.0 + ((g - Gc) / w) ** 4) - d0 + RHO * s / (KB + s)

def quasi_initials(x):
    meal, ps, g = np.asarray(x, dtype=float)[:3]
    s = ps / (SK * (1.0 + CG * g))
    c = sensitivity(s) * g
    y = meal + GREN - SG * g
    h = GHEP / (1.0 + ALPHAS * s)
    z = c - KI * y
    rad = z * z + 4.0 * c * KI * (y + h)
    if rad < 0:
        return np.full(4, np.nan)
    den = z + np.sqrt(rad)
    if abs(den) < 1e-14:
        return np.full(4, np.nan)
    I = 2.0 * (y + h) / den
    B = DI * I / secretion(g, s)
    return np.array([g, I, B, s], dtype=float)

def initials(x):
    a = np.asarray(x, dtype=float)
    if a.shape != (6,):
        raise ValueError('需要六个个体拟合参数 [Gmeal, PS, G0, I0, beta0, S0]')
    return a[2:6].copy()

def rhs(t, state, meal, ps, sh=SH):
    g, I, B, s = state
    return [meal + GREN + GHEP / ((1 + ALPHAS * s) * (1 + KI * I)) - SG * g - sensitivity(s) * I * g, secretion(g, s) * B - DI * I, growth(g, s, sh) * B, ps / (1 + CG * g) - SK * s]

def ode(x, sh, ages, extra=(), rtol=1e-05):
    a = np.asarray(ages, float)
    ext = np.asarray(extra, float)
    first = float(a.min())
    aa = np.sort(np.unique(np.r_[a, ext]))
    tt = (aa - first) * D
    y0 = initials(x)
    if not np.isfinite(y0).all() or y0.min() <= 0:
        return None
    if tt[-1] <= 0:
        return (aa, y0[:, None], None)
    try:
        z = solve_ivp(rhs, (0, float(tt[-1])), y0, t_eval=tt, args=(x[0], x[1], sh), method='LSODA', rtol=rtol, atol=[1e-05, 1e-08, 1e-13, 0.0002], max_step=120.0)
        if not z.success or not np.isfinite(z.y).all() or z.y.min() < -1e-06 or (z.y[0].max() > 650):
            return None
        return (aa, z.y, None)
    except Exception:
        return None

def load_reference():
    return pd.read_csv(StringIO(REFERENCE_FITTED_CSV)).set_index('subject')
LOWER = np.array([1285.0, 1800.0, 75.0, 0.02, 5e-05, 20.0])
UPPER = np.array([2357.0, 4500.0, 125.0, 12.0, 0.12, 1800.0])
OPT_LOWER = np.r_[LOWER[:3], [-0.2, -0.2, -0.2]]
OPT_UPPER = np.r_[UPPER[:3], [0.2, 0.2, 0.2]]
OPT_SCALE = np.array([500.0, 200.0, 12.0, 0.08, 0.08, 0.08])

def unpack_candidate(y):
    q = quasi_initials(y[:3])
    if not np.isfinite(q).all() or np.min(q) <= 0:
        return None
    x = np.r_[y[:3], q[1:] * np.exp(y[3:])]
    if np.any(x < LOWER) or np.any(x > UPPER):
        return None
    return x

def simple_pso(objective, initial_positions, niter=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(initial_positions)
    dim = len(OPT_LOWER)
    low, hi = (OPT_LOWER, OPT_UPPER)
    X = np.vstack(initial_positions)
    X = np.clip(X, low + 1e-07, hi - 1e-07)
    V = rng.uniform(-0.08, 0.08, (n, dim)) * (hi - low)
    score = np.array([objective(x) for x in X])
    local = X.copy()
    lscore = score.copy()
    best_i = int(np.argmin(lscore))
    best = local[best_i].copy()
    best_score = float(lscore[best_i])
    for it in range(niter):
        r1, r2 = rng.random((2, n, dim))
        V = 0.55 * V + 1.25 * r1 * (local - X) + 1.35 * r2 * (best - X)
        V = np.clip(V, -0.24 * (hi - low), 0.24 * (hi - low))
        X = np.clip(X + V, low + 1e-07, hi - 1e-07)
        score = np.array([objective(x) for x in X])
        improved = score < lscore
        local[improved] = X[improved]
        lscore[improved] = score[improved]
        best_i = int(np.argmin(lscore))
        if lscore[best_i] < best_score:
            best, best_score = (local[best_i].copy(), float(lscore[best_i]))
    return (best, best_score)

def fit_subject(sid, sh, obs_table, reference, trials=4, nfev=75, pso_iters=10):
    df = obs_table.loc[obs_table.subject == sid].sort_values('year')
    a, observed = (df.year.to_numpy(), df.glucose.to_numpy())
    prior = reference.loc[sid]
    old3 = np.array([prior.Gmeal, prior.PS, prior.G0])
    q0 = quasi_initials(old3)
    if not np.isfinite(q0).all() or np.min(q0) <= 0:
        raise RuntimeError(f'Subject {sid}: invalid reference quasi equilibrium')
    nres = len(a) + 4

    def residual(y):
        x = unpack_candidate(y)
        if x is None:
            return np.full(nres, 300.0)
        sol = ode(x, sh, a, rtol=3e-06)
        if sol is None:
            return np.full(nres, 300.0)
        ages, Y, _ = sol
        return np.r_[np.interp(a, ages, Y[0]) - observed, 0.025 * (x[2] - observed[0]), QSS_LOG_PENALTY * y[3:]]

    def scalar(y):
        z = residual(y)
        return float(z @ z)
    islam_meal, islam_g = ISLAM_GMEAL_G0[sid]
    islam3 = np.array([islam_meal, old3[1], islam_g])
    islam3 = np.clip(islam3, LOWER[:3] + 1e-05, UPPER[:3] - 1e-05)
    sixrow = smooth6_seed().loc[sid]
    old6 = np.array([sixrow.Gmeal, sixrow.PS, sixrow.G0, sixrow.I0, sixrow.beta0, sixrow.S0])
    q6 = quasi_initials(old6[:3])
    smooth_start = np.r_[old6[:3], np.log(old6[3:] / q6[1:])]
    rng = np.random.default_rng(2025 + sid)
    particles = [np.r_[old3, [0.0, 0.0, 0.0]], np.r_[islam3, [0.0, 0.0, 0.0]], smooth_start]
    for _ in range(8):
        v = np.r_[rng.uniform(LOWER[:2], UPPER[:2]), rng.uniform(max(LOWER[2], observed[0] - 8), min(UPPER[2], observed[0] + 8)), rng.normal(0, 0.04, 3)]
        particles.append(v)
    best_y, _ = simple_pso(scalar, particles, niter=pso_iters, seed=305 + sid)
    local_starts = [smooth_start, best_y, np.r_[old3, [0.0, 0.0, 0.0]], np.r_[islam3, [0.0, 0.0, 0.0]]]
    best = (np.inf, None)
    for j, st in enumerate(local_starts[:trials]):
        try:
            res = least_squares(residual, np.clip(st, OPT_LOWER + 1e-07, OPT_UPPER - 1e-07), bounds=(OPT_LOWER, OPT_UPPER), x_scale=OPT_SCALE, max_nfev=nfev, ftol=2e-06, xtol=2e-06, gtol=2e-06, diff_step=5e-05)
            val = scalar(res.x)
            if val < best[0]:
                best = (val, res.x.copy())
        except Exception as ex:
            print(f'Subject {sid}: local start {j + 1} failed: {ex}', flush=True)
    if best[1] is None:
        raise RuntimeError(f'Subject {sid}: all fits failed')
    y = best[1]
    x = unpack_candidate(y)
    z = ode(x, sh, a, rtol=3e-07)
    if z is None:
        raise RuntimeError(f'Subject {sid}: final integration failed')
    ages, Y, _ = z
    predicted = np.interp(a, ages, Y[0])
    error = predicted - observed
    q = quasi_initials(x[:3])
    first_week = solve_ivp(rhs, (0, 7), x[2:6], args=(x[0], x[1], sh), method='LSODA', t_eval=[0, 1, 7], rtol=3e-08, atol=[1e-07, 1e-09, 1e-14, 1e-06])
    gday = float(first_week.y[0, 1] - x[2]) if first_week.success else np.nan
    iday = float(first_week.y[1, 1] - x[3]) if first_week.success else np.nan
    row = dict(subject=sid, Gmeal=x[0], PS=x[1], G0=x[2], I0=x[3], beta0=x[4], S0=x[5], RMSE=float(np.sqrt(np.mean(error ** 2))), baseline3_RMSE=float(prior.RMSE), I0_quasi=q[1], beta0_quasi=q[2], S0_quasi=q[3], I0_ratio=x[3] / q[1], beta0_ratio=x[4] / q[2], S0_ratio=x[5] / q[3], G_change_first_day=gday, I_change_first_day=iday)
    preds = pd.DataFrame(dict(subject=sid, age=a, observed=observed, predicted=predicted, residual=error))
    return (row, preds)

def _times_font():
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in ('Times New Roman', 'Times', 'Tinos', 'Liberation Serif'):
        if candidate in installed:
            return candidate
    return 'serif'
PLOT_FONT = _times_font()
BASE_FONT_SIZE = 25
LEGEND_FONT_SIZE = 18

def configure_plotting():
    plt.rcParams.update({'font.family': PLOT_FONT, 'font.size': BASE_FONT_SIZE, 'axes.titlesize': BASE_FONT_SIZE, 'axes.labelsize': BASE_FONT_SIZE, 'xtick.labelsize': BASE_FONT_SIZE, 'ytick.labelsize': BASE_FONT_SIZE, 'legend.fontsize': LEGEND_FONT_SIZE, 'figure.titlesize': BASE_FONT_SIZE, 'mathtext.fontset': 'custom', 'mathtext.rm': PLOT_FONT, 'mathtext.it': PLOT_FONT + ':italic', 'mathtext.bf': PLOT_FONT + ':bold', 'axes.spines.top': True, 'axes.spines.right': True, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.75, 'xtick.direction': 'in', 'ytick.direction': 'in'})

def fit_cohort_styled(obs_table, *, trials=4, nfev=75, pso_iters=10):
    ref = load_reference()
    rows, preds = ([], [])
    for sid in sorted(obs_table.subject.unique()):
        row, df = fit_subject(sid, SH, obs_table, ref, trials, nfev, pso_iters)
        rows.append(row)
        preds.append(df)
        print(f"Subject {sid:02d}: RMSE {row['RMSE']:.3f} mg/dL, G0={row['G0']:.2f}, I0={row['I0']:.3f}, beta0={row['beta0']:.5f}, S0={row['S0']:.2f}", flush=True)
    params = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True)
    p11 = params.loc[params.subject == 11].iloc[0]
    ob11 = obs_table.loc[obs_table.subject == 11].sort_values('year')
    x = np.array([p11.Gmeal, p11.PS, p11.G0, p11.I0, p11.beta0, p11.S0])
    start_age = float(ob11.year.min())
    grid = np.linspace(start_age, 85.0, 600)
    sim = solve_ivp(rhs, (0.0, (85.0 - start_age) * D), initials(x), t_eval=(grid - start_age) * D, args=(x[0], x[1], SH), method='LSODA', rtol=5e-08, atol=[1e-06, 1e-09, 1e-13, 1e-06], max_step=120.0)
    if sim.success:
        out = dict(subject=11, age_start=start_age, age_end=85.0, glucose_age_85=float(sim.y[0, -1]), max_glucose_to_85=float(np.max(sim.y[0])), stays_below_126=bool(np.max(sim.y[0]) < 126), warning='Conditional extrapolation only; not clinical proof.')
    else:
        out = dict(subject=11, age_start=start_age, age_end=85.0, glucose_age_85=np.nan, max_glucose_to_85=np.nan, stays_below_126=False, warning=f'ODE failed: {sim.message}')
    print('Subject 11 nonprogression audit:', out, flush=True)
    return (params, predictions)
C_G = '#C65D65'
C_S = '#C48A3A'
C_I = '#168A83'
C_B = '#8064A8'
C_SI = '#2B62B8'
C_OBS = '#262626'
TRACE_LW = 2.8
FORECAST_DASHES = (5.0, 3.0)
THRESHOLD_G = 126.0
PLOT_GRID = '#E7ECF0'
PROJECTION_SHADE = '#EEF2F5'

def stylize(ax):
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=PLOT_GRID, lw=0.75)
    for name, spine in ax.spines.items():
        spine.set_linewidth(0.85)
        spine.set_color('#A7B0B9')
        spine.set_visible(name not in ('top',))
    ax.tick_params(direction='out', length=4, width=0.75, color='#71808B', labelcolor='#263644', labelsize=BASE_FONT_SIZE, top=False)

def _plot_with_projection(ax, ages, vals, observed_last, color, *, zorder=4):
    ages = np.asarray(ages, float)
    vals = np.asarray(vals, float)
    if observed_last >= ages[-1]:
        ax.plot(ages, vals, color=color, lw=TRACE_LW, zorder=zorder)
        return
    if observed_last <= ages[0]:
        ax.plot(ages, vals, color=color, lw=TRACE_LW, ls='--', zorder=zorder)
        return
    v_at_last = np.interp(observed_last, ages, vals)
    before = ages < observed_last
    after = ages > observed_last
    ax.plot(np.r_[ages[before], observed_last], np.r_[vals[before], v_at_last], color=color, lw=TRACE_LW, zorder=zorder)
    ax.plot(np.r_[observed_last, ages[after]], np.r_[v_at_last, vals[after]], color=color, lw=TRACE_LW, ls=(0, FORECAST_DASHES), zorder=zorder)

def _mark_extrapolation(ax_top, ax_bottom, observed_last, end):
    if observed_last >= end:
        return
    for ax in (ax_top, ax_bottom):
        ax.axvspan(observed_last, end, facecolor='#F4F6F8', zorder=0, lw=0)
        ax.axvline(observed_last, color='#8998A5', lw=1.1, ls=(0, (2, 3)), zorder=2)
    ax_top.text(0.97, 0.975, 'Model projection', transform=ax_top.transAxes, ha='right', va='top', fontsize=14, color='#687684', fontstyle='italic', zorder=8)

def _long_solution(sid, params, obs_table, end, n=2400):
    p = params.loc[sid]
    obs = obs_table.loc[obs_table.subject == sid].sort_values('year')
    start = float(obs.year.min())
    x = np.array([p.Gmeal, p.PS, p.G0, p.I0, p.beta0, p.S0])
    z = solve_ivp(rhs, (0, (end - start) * D), initials(x), args=(p.Gmeal, p.PS, SH), method='LSODA', rtol=5e-08, atol=[1e-07, 1e-10, 1e-14, 5e-06], max_step=50.0, dense_output=True)
    if not z.success:
        raise RuntimeError(f'Subject {sid}: {z.message}')
    ages = np.linspace(start, end, n)
    G, I, B, S = z.sol((ages - start) * D)
    return (obs, dict(age=ages, G=G, I=I, beta=B, S=S, SI=sensitivity(S)))

def _model_onset(ages, glucose):
    idx = np.flatnonzero((glucose[1:] >= THRESHOLD_G) & (glucose[:-1] < THRESHOLD_G))
    if not len(idx):
        return None
    j = int(idx[0])
    a, b = (glucose[j], glucose[j + 1])
    return float(ages[j] + (THRESHOLD_G - a) / (b - a) * (ages[j + 1] - ages[j]))

def _patient_range(sid, params, obs_table):
    ob = obs_table.loc[obs_table.subject == sid].sort_values('year')
    begin = float(ob.year.min())
    last = float(ob.year.max())
    if sid == 11:
        return (85.0, None)
    search_end = max(90.0, last + 35.0)
    _, full = _long_solution(sid, params, obs_table, search_end, 1700)
    onset = _model_onset(full['age'], full['G'])
    end = max(last + 4.0, onset + 7.0 if onset is not None else last + 7.0)
    end = min(end, search_end)
    return (float(end), onset)

def _shade_both(ax1, ax2, onset, begin, end):
    if onset is None or not begin < onset < end:
        return
    for ax in (ax1, ax2):
        ax.axvline(onset, color='#778899', lw=1.25, ls=(0, (3, 3)), zorder=2)
    ax1.axhline(THRESHOLD_G, color='#AAB7C2', lw=1.0, ls=(0, (2, 4)), zorder=2)

def place_legend_inside(ax, handles, curves, ncol=2, avoid=()):
    legend = ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=ncol, frameon=True, fontsize=LEGEND_FONT_SIZE, facecolor='white', edgecolor='#D8E0E6', handlelength=1.8, handletextpad=0.5, columnspacing=0.8, labelspacing=0.16, borderpad=0.3, borderaxespad=0.1)
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.6)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer=renderer)
    bbox = legend.get_window_extent(renderer=renderer)
    bw, bh = (bbox.width, bbox.height)
    fw, fh = (frame.width, frame.height)
    if bw >= fw - 10 or bh >= fh - 10:
        legend.set_fontsize(max(8, LEGEND_FONT_SIZE - 2))
        fig.canvas.draw()
        bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
        bw, bh = (bbox.width, bbox.height)
    points = []
    for line_ax, xv, yv, weight in curves:
        xv = np.asarray(xv, dtype=float)
        yv = np.asarray(yv, dtype=float)
        jj = np.linspace(0, len(xv) - 1, min(len(xv), 750)).astype(int)
        xy = np.column_stack((xv[jj], yv[jj]))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            pixel_xy = line_ax.transData.transform(xy)
            points.append((pixel_xy, weight))
    margin = 5.0
    xmin = (bw / 2 + margin) / fw
    xmax = 1 - xmin
    ymin = (bh / 2 + margin) / fh
    ymax = 1 - ymin
    xs = np.linspace(xmin, xmax, 15) if xmin < xmax else [0.5]
    ys = np.linspace(ymin, ymax, 17) if ymin < ymax else [0.5]
    choice = (float('inf'), 0.5, 0.5)
    for cy in ys:
        for cx in xs:
            xlo = frame.x0 + cx * fw - bw / 2
            xhi = xlo + bw
            ylo = frame.y0 + cy * fh - bh / 2
            yhi = ylo + bh
            score = 0.0
            clearance = 7.0
            for pts, wt in points:
                inside = (pts[:, 0] >= xlo - clearance) & (pts[:, 0] <= xhi + clearance) & (pts[:, 1] >= ylo - clearance) & (pts[:, 1] <= yhi + clearance)
                score += wt * inside.sum() / max(1, len(pts))
            for bx, by in avoid:
                px = frame.x0 + bx * fw
                py = frame.y0 + by * fh
                if xlo - 15 <= px <= xhi + 15 and ylo - 15 <= py <= yhi + 15:
                    score += 4
            score += 0.001 * ((cx - 0.22) ** 2 + (cy - 0.83) ** 2)
            if score < choice[0]:
                choice = (score, float(cx), float(cy))
    legend.set_bbox_to_anchor((choice[1], choice[2]), transform=ax.transAxes)
    return legend

def tidy_y_axis(ax, min_step=0.0):
    """Use evenly spaced, rounded y-ticks, without offset/scientific labels.

    The right-side insulin-sensitivity scale needs at least 0.01 increments;
    otherwise tiny variation can produce distracting 4-decimal tick labels.
    """
    from matplotlib.ticker import FormatStrFormatter

    low, high = ax.get_ylim()
    requested = max((high - low) / 6.0, min_step)
    order = 10.0 ** np.floor(np.log10(requested))
    step = next(v * order for v in (1.0, 2.0, 2.5, 5.0, 10.0)
                if v * order >= requested * (1.0 - 1e-12))
    low = np.floor(low / step + 1e-10) * step
    high = np.ceil(high / step - 1e-10) * step
    ticks = np.arange(round(low / step), round(high / step) + 1) * step
    ndigits = max(0, len(f'{step:.10f}'.rstrip('0').split('.')[-1]) if step < 1 else 0)
    ax.set_ylim(float(low), float(high))
    ax.set_yticks(np.round(ticks, 10))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{ndigits}f'))


def tidy_all_y_axes(ax_top, top2, ax_bottom, bottom2):
    tidy_y_axis(ax_top)
    tidy_y_axis(top2)
    tidy_y_axis(ax_bottom)
    tidy_y_axis(bottom2, min_step=0.01)


def draw_patient_pair(ax_top, ax_bottom, sid, params, obs_table, *, standalone=False):
    end, onset = _patient_range(sid, params, obs_table)
    obs, t = _long_solution(sid, params, obs_table, end, 2400)
    age = t['age']
    start = float(obs.year.min())
    last = float(obs.year.max())
    stylize(ax_top)
    stylize(ax_bottom)
    _mark_extrapolation(ax_top, ax_bottom, last, end)
    _shade_both(ax_top, ax_bottom, onset, start, end)
    top2 = ax_top.twinx()
    bottom2 = ax_bottom.twinx()
    stylize(top2)
    stylize(bottom2)
    _plot_with_projection(ax_top, age, t['G'], last, C_G, zorder=4)
    _plot_with_projection(ax_top, age, t['S'], last, C_S, zorder=4)
    ax_top.plot(obs.year, obs.glucose, marker='D', ms=7.5, mfc='white', mec=C_OBS, mew=1.4, ls='none', zorder=8)
    _plot_with_projection(top2, age, t['I'], last, C_I, zorder=5)
    _plot_with_projection(ax_bottom, age, t['beta'], last, C_B, zorder=4)
    _plot_with_projection(bottom2, age, t['SI'], last, C_SI, zorder=5)
    ax_top.set_xlim(start, end)
    ax_bottom.set_xlim(start, end)
    ax_top.set_ylim(0, max(500.0, 1.08 * float(max(t['G'].max(), t['S'].max()))))
    top2.set_ylim(0, max(0.4, np.ceil(1.15 * float(t['I'].max()) * 2) / 2))
    ax_bottom.set_ylim(0, max(0.002, 1.1 * float(np.nanmax(t['beta']))))
    vmin = float(np.nanmin(t['SI']))
    vmax = float(np.nanmax(t['SI']))
    d = max(vmax - vmin, 0.015)
    bottom2.set_ylim(max(0, vmin - 0.12 * d), vmax + 0.25 * d)
    tidy_all_y_axes(ax_top, top2, ax_bottom, bottom2)
    ax_top.set_ylabel('G, S', fontsize=BASE_FONT_SIZE, labelpad=12)
    top2.set_ylabel('Insulin', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_bottom.set_ylabel('$\\beta$', fontsize=BASE_FONT_SIZE, labelpad=12)
    bottom2.set_ylabel('$S_I$', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_bottom.set_xlabel('Time (years)', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(f'Subject {sid}', fontsize=BASE_FONT_SIZE, fontweight='semibold', pad=18, color='#24384A')
    from matplotlib.lines import Line2D
    legend_top = [Line2D([0], [0], color=C_G, lw=TRACE_LW, label='G (mg/dL)'), Line2D([0], [0], color=C_S, lw=TRACE_LW, label='SCFAs (μmol)'), Line2D([0], [0], marker='D', mfc='white', mec=C_OBS, ls='none', markersize=6.0, label='Observed G (mg/dL)'), Line2D([0], [0], color=C_I, lw=TRACE_LW, label='I (μU/mL)')]
    legend_bottom = [Line2D([0], [0], color=C_B, lw=TRACE_LW, label='$\\beta$ cell functional mass (a.u.)'), Line2D([0], [0], color=C_SI, lw=TRACE_LW, label='Insulin sensitivity $S_I$ (mL/(μU·day))')]
    place_legend_inside(ax_top, legend_top, [(ax_top, age, t['G'], 1.0), (ax_top, age, t['S'], 1.0), (top2, age, t['I'], 1.0), (ax_top, obs.year, obs.glucose, 3.0)], ncol=1, avoid=([(0.9, 0.11)] if onset is not None else []) + ([(0.9, 0.91)] if standalone else []))
    place_legend_inside(ax_bottom, legend_bottom, [(ax_bottom, age, t['beta'], 1.0), (bottom2, age, t['SI'], 1.0)], ncol=1)
    return onset
NORMAL_GMEAL = 1310.0
NORMAL_PS = 2350.0
NORMAL_G0 = 90.0
NORMAL_START_AGE = 10.0
NORMAL_END_AGE = 85.0

def draw_normal_scenario(ax_top, ax_bottom):
    start, end = (NORMAL_START_AGE, NORMAL_END_AGE)
    y0 = quasi_initials([NORMAL_GMEAL, NORMAL_PS, NORMAL_G0])
    if not np.isfinite(y0).all() or np.min(y0) <= 0:
        raise ValueError('Normal scenario: invalid initial state')
    ages = np.linspace(start, end, 2400)
    z = solve_ivp(rhs, (0.0, (end - start) * D), y0, args=(NORMAL_GMEAL, NORMAL_PS, SH), method='LSODA', t_eval=(ages - start) * D, rtol=5e-08, atol=[1e-07, 1e-10, 1e-14, 5e-06], max_step=50.0)
    if not z.success:
        raise RuntimeError('Normal scenario integration failed: ' + z.message)
    G, I, B, S = z.y
    SI = sensitivity(S)
    stylize(ax_top)
    stylize(ax_bottom)
    top2 = ax_top.twinx()
    bottom2 = ax_bottom.twinx()
    stylize(top2)
    stylize(bottom2)
    ax_top.plot(ages, G, c=C_G, lw=TRACE_LW, zorder=4)
    ax_top.plot(ages, S, c=C_S, lw=TRACE_LW, zorder=4)
    top2.plot(ages, I, c=C_I, lw=TRACE_LW, zorder=5)
    ax_bottom.plot(ages, B, c=C_B, lw=TRACE_LW, zorder=4)
    bottom2.plot(ages, SI, c=C_SI, lw=TRACE_LW, zorder=5)
    ax_top.set_xlim(start, end)
    ax_bottom.set_xlim(start, end)
    ax_top.set_ylim(0, max(500.0, 1.08 * float(max(G.max(), S.max()))))
    top2.set_ylim(0, max(0.4, np.ceil(1.15 * float(I.max()) * 2) / 2))
    ax_bottom.set_ylim(0, max(0.002, 1.1 * float(B.max())))
    vmin, vmax = (float(SI.min()), float(SI.max()))
    delta = max(vmax - vmin, 0.015)
    bottom2.set_ylim(max(0.0, vmin - 0.12 * delta), vmax + 0.25 * delta)
    tidy_all_y_axes(ax_top, top2, ax_bottom, bottom2)
    ax_top.set_ylabel('G, S', fontsize=BASE_FONT_SIZE, labelpad=12)
    top2.set_ylabel('Insulin', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_bottom.set_ylabel('$\\beta$', fontsize=BASE_FONT_SIZE, labelpad=12)
    bottom2.set_ylabel('$S_I$', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_bottom.set_xlabel('Time (years)', fontsize=BASE_FONT_SIZE, labelpad=12)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title('Normal metabolic scenario', fontsize=BASE_FONT_SIZE, fontweight='semibold', pad=18, color='#24384A')
    from matplotlib.lines import Line2D
    legend_top = [Line2D([0], [0], color=C_G, lw=TRACE_LW, label='G (mg/dL)'), Line2D([0], [0], color=C_S, lw=TRACE_LW, label='SCFAs (μmol)'), Line2D([0], [0], color=C_I, lw=TRACE_LW, label='I (μU/mL)')]
    legend_bottom = [Line2D([0], [0], color=C_B, lw=TRACE_LW, label='$\\beta$ cell functional mass (a.u.)'), Line2D([0], [0], color=C_SI, lw=TRACE_LW, label='Insulin sensitivity $S_I$ (mL/(μU·day))')]
    place_legend_inside(ax_top, legend_top, [(ax_top, ages, G, 1.0), (ax_top, ages, S, 1.0), (top2, ages, I, 1.0)], ncol=1)
    place_legend_inside(ax_bottom, legend_bottom, [(ax_bottom, ages, B, 1.0), (bottom2, ages, SI, 1.0)], ncol=1)
    return dict(G0=float(G[0]), G_end=float(G[-1]), I0=float(I[0]), beta0=float(B[0]), S0=float(S[0]), Gmeal=NORMAL_GMEAL, PS=NORMAL_PS)

def generate_normal_vs_subject11(params, obs_table):
    fig = plt.figure(figsize=(26.0, 11.5), facecolor='white')
    outer = fig.add_gridspec(1, 2, left=0.075, right=0.925, top=0.91, bottom=0.11, wspace=0.3)
    axes = []
    for col in range(2):
        inner = outer[0, col].subgridspec(2, 1, hspace=0.065)
        axes.append((fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0])))
    normal_info = draw_normal_scenario(*axes[0])
    draw_patient_pair(*axes[1], 11, params, obs_table)
    fig.savefig(OUT / 'normal-11.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'normal-11.eps', format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"Normal scenario (illustrative, not clinical): G0={normal_info['G0']:.2f}, G_end={normal_info['G_end']:.2f} mg/dL, Gmeal={NORMAL_GMEAL:.0f}, PS={NORMAL_PS:.0f}")
    print('Generated: normal-11.pdf/.eps')

def generate_publication_figures(params_df, obs_table):
    """Save each subject separately and Subjects 1/11, 5/6 as two-column pairs.

    All original plotting routines, model parameters and plot styles are reused.
    """
    configure_plotting()
    OUT.mkdir(parents=True, exist_ok=True)
    params = params_df.set_index('subject')

    # One subject per figure, using the same two vertically stacked panels.
    # The plotting area retains approximately the same width as one column
    # in the original 26 x 9.5-inch two-subject figures.
    for sid in sorted(params.index):
        fig = plt.figure(figsize=(14.0, 9.5), facecolor='white')
        inner = fig.add_gridspec(2, 1, left=0.13, right=0.85,
                                 top=0.91, bottom=0.095, hspace=0.09)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bottom = fig.add_subplot(inner[1, 0])
        draw_patient_pair(ax_top, ax_bottom, sid, params, obs_table)
        name = f'subject{sid}'
        fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(OUT / f'{name}.eps', format='eps', bbox_inches='tight')
        plt.close(fig)
        print(f'Generated {name}.pdf/.eps', flush=True)

    # Preserve the exact two-column figure size, grids and plotting method
    # from the user's original combined-subject plots.
    for subjects in ((1, 11), (5, 6)):
        fig = plt.figure(figsize=(26.0, 9.5), facecolor='white')
        outer = fig.add_gridspec(1, 2, left=0.075, right=0.925,
                                 top=0.91, bottom=0.095, wspace=0.3)
        for col, sid in enumerate(subjects):
            inner = outer[0, col].subgridspec(2, 1, hspace=0.09)
            ax_top = fig.add_subplot(inner[0, 0])
            ax_bottom = fig.add_subplot(inner[1, 0])
            draw_patient_pair(ax_top, ax_bottom, sid, params, obs_table)
        name = f'subject{subjects[0]}-{subjects[1]}'
        fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(OUT / f'{name}.eps', format='eps', bbox_inches='tight')
        plt.close(fig)
        print(f'Generated {name}.pdf/.eps', flush=True)

def main():
    parser = argparse.ArgumentParser(description='Unified PSO–TRF fit for 11 subjects')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--nfev', type=int, default=75)
    parser.add_argument('--pso-iters', type=int, default=10)
    args = parser.parse_args()
    obs = load_observations()
    if args.plot_only:
        params = pd.read_csv(P / 'SCFA_6param_Islam_style_parameters.csv')
        preds = pd.read_csv(P / 'SCFA_6param_Islam_style_predictions.csv')
    else:
        params, preds = fit_cohort_styled(obs, trials=args.trials, nfev=args.nfev, pso_iters=args.pso_iters)
        err = preds.predicted - preds.observed
        print(f'RMSE={np.sqrt(np.mean(err ** 2)):.4f} mg/dL')
    if not args.no_plots:
        generate_publication_figures(params, obs)
    print("\n11名患者的拟合参数：")
    cols = ['subject', 'Gmeal', 'PS', 'G0', 'I0', 'beta0', 'S0']
    print(params[cols].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
if __name__ == '__main__':
    main()
