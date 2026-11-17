import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

# ------------------------- Fixed model and data -------------------------
DAYS_PER_YEAR = 365.25
FIRST_AGE = 11.31
OBS_AGES = np.array([11.31,13.358,15.554,17.473,22.045,27.601,
                     30.801,33.539,36.286,42.068,44.132,45.812])
OBS_G = np.array([88.972,91.961,86.920,86.633,90.524,87.273,
                  93.605,92.436,104.117,103.611,120.687,177.592])

# Previously fitted Subject 1: Gmeal, PS, G0, I0, beta0, S0.
FIT = np.array([1381.85073846983,2331.1278115664,87.0704165772,
                2.52119728566847,0.0200376134221957,441.477836884565])
GMEAL, PS = FIT[:2]
Y0 = FIT[2:].copy()  # [G, I, beta, S]

# Unchanged shared coefficients from the original four-dimensional ODE.
SG, GREN, GHEP = 8., 107., 535.
ALPHAS, KI, SI0, ETAS, KS = .0043, .05, .635, 7.2, 100.
AG, SIGMA_G, GAMMA_S, KGLP, DI = 20000., 86400., .1, 100., 200.
KB, SK, CG, RHO = 100., 4., .0037, .047
D0, A, GC, W = .0402, .0021, 103., 17.5
THRESHOLDS = {'early':100., 'late':126.}
STRATEGIES = {
    'Control': (1.,1.),
    'PS+20%': (1.20,1.),
    'Gmeal-10%': (1.,.90),
    'Combined+20%': (1.20,.90),
}
FIGURES = [
    ('Figure1_SCFA_intervention_G', 'Effects of SCFA supply intervention',
     'PS+20%', r'$P_S$ +20%', '#4776B4'),
    ('Figure2_Gmeal_intervention_G', 'Effects of dietary glucose input reduction',
     'Gmeal-10%', r'$G_{\mathrm{meal}}$ -10%', '#C48A3A'),
    ('Figure3_combined_intervention_G', 'Effects of combined intervention',
     'Combined+20%', r'$P_S$ +20%, $G_{\mathrm{meal}}$ -10%', '#8064A8'),
]

RED = '#C65D65'
OBS_COLOR = '#262626'
DASHES = (5.,3.)


def rhs(t, state, meal, ps):
    """Original four-dimensional G-I-beta-S model, time in days."""
    g, I, beta, s = state
    sensitivity = SI0 * (1. + ETAS*s/(KS+s))
    secretion = SIGMA_G*g*g/(AG+g*g) * (1. + GAMMA_S*s/(KGLP+s))
    growth = A/(1. + ((g-GC)/W)**4) - D0 + RHO*s/(KB+s)
    return [meal + GREN + GHEP/((1.+ALPHAS*s)*(1.+KI*I)) - SG*g - sensitivity*I*g,
            secretion*beta - DI*I, growth*beta,
            ps/(1.+CG*g) - SK*s]


def integrate_piece(y0, start, end, meal, ps, n):
    """Integrate one constant-input segment, return age, states, dense solution."""
    if end <= start:
        raise ValueError('Invalid integration interval')
    duration = (end-start)*DAYS_PER_YEAR
    sol = solve_ivp(rhs, (0.,duration), y0, args=(meal,ps),
                    method='LSODA', dense_output=True,
                    rtol=2e-8, atol=[1e-7,1e-10,1e-14,5e-6], max_step=50.)
    if not sol.success:
        raise RuntimeError(sol.message)
    ages = np.linspace(start,end,max(2,int(n)))
    states = sol.sol((ages-start)*DAYS_PER_YEAR)
    if not np.isfinite(states).all():
        raise RuntimeError('ODE returned non-finite values')
    return ages,states,sol.sol


def first_crossing(ages, glucose, level, dense):
    """First *upward* threshold crossing on the untreated trajectory."""
    indices = np.flatnonzero((glucose[:-1]<level)&(glucose[1:]>=level))
    if not len(indices):
        raise RuntimeError(f'Untreated model never reaches {level} mg/dL')
    j = int(indices[0])
    return float(brentq(lambda age: dense((age-FIRST_AGE)*DAYS_PER_YEAR)[0]-level,
                        ages[j],ages[j+1],xtol=1e-9))


def simulate(end_age, n_grid, trigger=np.nan, ps_factor=1., meal_factor=1.):
    """Apply intervention as a step change, preserving continuous ODE states."""
    if not np.isfinite(trigger):
        age,states,_ = integrate_piece(Y0,FIRST_AGE,end_age,GMEAL,PS,n_grid)
        return age,states
    n_pre = max(30,round(n_grid*(trigger-FIRST_AGE)/(end_age-FIRST_AGE)))
    n_post = max(30,n_grid-n_pre+1)
    pre_age,pre,_ = integrate_piece(Y0,FIRST_AGE,trigger,GMEAL,PS,n_pre)
    post_age,post,_ = integrate_piece(pre[:,-1],trigger,end_age,
                                      GMEAL*meal_factor,PS*ps_factor,n_post)
    return np.r_[pre_age,post_age[1:]],np.column_stack((pre,post[:,1:]))


def stabilized_age(age, glucose, min_age, tolerance=1., followup=3.):
    """First time after which G remains within tolerance of its age-85 value."""
    tail_dev = np.maximum.accumulate(np.abs(glucose-glucose[-1])[::-1])[::-1]
    valid = np.flatnonzero((age>=min_age-1e-8)&(age<=age[-1]-followup)
                               &(tail_dev<=tolerance))
    return float(age[valid[0]]) if len(valid) else np.nan


def clip(age, vals, end):
    mask = age<end
    return np.r_[age[mask],end],np.r_[vals[mask],np.interp(end,age,vals)]


def plotting_style():
    installed = {x.name for x in font_manager.fontManager.ttflist}
    font = next((f for f in ('Times New Roman','Times','Tinos','Liberation Serif')
                 if f in installed),'DejaVu Serif')
    plt.rcParams.update({'font.family':font, 'mathtext.fontset':'stix',
                         'pdf.fonttype':42, 'ps.fonttype':42})


def draw_figure(filename, heading, key, description, color, trajectories,
                crossing, end_age, tolerance, out):
    """Early/late panels; only G is plotted, legends stay inside."""
    stability = [stabilized_age(trajectories[stage][k][0],
                 trajectories[stage][k][1][0],max(OBS_AGES[-1],crossing[stage]),
                 tolerance) for stage in THRESHOLDS for k in ('Control',key)]
    if not np.isfinite(stability).all():
        raise RuntimeError(f'{filename}: G not stabilized by age {end_age}; '
                           'increase --end-age rather than truncating arbitrarily')
    cutoff = max(stability)+3.
    if cutoff>end_age:
        raise RuntimeError('Not enough time after stabilization')
    ymax = max(250., np.ceil((max(trajectories[stage][k][1][0].max()
                for stage in THRESHOLDS for k in ('Control',key))+7.)/25.)*25.)
    fig,axs = plt.subplots(1,2,figsize=(17.8,6.2),sharex=True,sharey=True)

    for ax,(stage,threshold) in zip(axs,THRESHOLDS.items()):
        trigger = crossing[stage]
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E7ECF0',lw=.75)

        for name,spine in ax.spines.items():
            spine.set_linewidth(.85)
            spine.set_color('#A7B0B9')
            spine.set_visible(name!='top')

        ax.tick_params(direction='out',length=4,width=.75,labelsize=16)

        ax.axvspan(OBS_AGES[-1]-FIRST_AGE,cutoff-FIRST_AGE,
                   facecolor='#EEF2F5',zorder=0)
        ax.axvline(OBS_AGES[-1]-FIRST_AGE,color='#A0ACB6',lw=1.,ls=(0,(2,3)))
        ax.axvline(trigger-FIRST_AGE,color='#5E7284',lw=1.35,ls=':')
        ax.axhline(threshold,color='#BBC5CF',lw=.9,ls=(0,(2,4)))

        age,y = trajectories[stage]['Control']
        age,g = clip(age,y[0],cutoff)
        ax.plot(age-FIRST_AGE,g,color=RED,lw=2.9,ls='-',zorder=4)

        age,y = trajectories[stage][key]
        age,g = clip(age,y[0],cutoff)
        after = age>=trigger-1e-7
        ax.plot(age[after]-FIRST_AGE,g[after],color=color,lw=2.8,
                ls=(0,DASHES),zorder=5)

        ax.plot(OBS_AGES-FIRST_AGE,OBS_G,marker='D',ms=7.5,
                mfc='white',mec=OBS_COLOR,mew=1.4,ls='none',zorder=9)

        ax.set_xlim(0,cutoff-FIRST_AGE)
        ax.set_ylim(65,ymax)
        ax.set_xlabel('Time (years)',fontsize=24,labelpad=10)
        ax.set_ylabel(r'Glucose $G$ (mg/dL)',fontsize=24,labelpad=10)
        ax.set_xticks(np.arange(0,cutoff-FIRST_AGE+.01,
                    5. if cutoff-FIRST_AGE<=52. else 10.))

        handles = [
            Line2D([0],[0],color=RED,lw=2.9,ls='-',label='Baseline'),
            Line2D([0],[0],color=color,lw=2.8,ls=(0,DASHES),label=description),
            Line2D([0],[0],marker='D',mfc='white',mec=OBS_COLOR,
                   ls='none',ms=7,label='Observed glucose')
        ]

        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.025,.975),
                  fontsize=14,facecolor='white',framealpha=1.,
                  edgecolor='#D5DDE4',handlelength=2.8,labelspacing=.38,borderpad=.5)

    fig.subplots_adjust(left=.082,right=.977,bottom=.14,top=.94,wspace=.17)

    for suffix in ('png','eps'):
        kw = {'dpi':230} if suffix=='png' else {}
        fig.savefig(out/f'{filename}.{suffix}',bbox_inches='tight',**kw)

    plt.close(fig)

    return dict(
        figure=filename,
        latest_stability_time_years=max(stability)-FIRST_AGE,
        figure_end_time_years=cutoff-FIRST_AGE,
        early_trigger_time_years=crossing['early']-FIRST_AGE,
        late_trigger_time_years=crossing['late']-FIRST_AGE
    )


def main():
    parser = argparse.ArgumentParser(description='Subject 1 glucose intervention figures only')
    parser.add_argument('--out',default='SCFA_subject1_intervention_figures')
    parser.add_argument('--end-age',type=float,default=85.)
    parser.add_argument('--n-grid',type=int,default=2200)
    parser.add_argument('--stable-tolerance',type=float,default=1.)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)

    if args.end_age<=OBS_AGES[-1]+3:
        parser.error('--end-age must exceed the final observation age by 3 years')

    plotting_style()

    # Baseline; crossing times always computed from untreated G(t).
    age0,y0,dense = integrate_piece(
        Y0,FIRST_AGE,args.end_age,GMEAL,PS,args.n_grid
    )

    crossing = {
        name:first_crossing(age0,y0[0],value,dense)
        for name,value in THRESHOLDS.items()
    }

    print('Intervention starts (years since first observation):',
          {k:round(v-FIRST_AGE,3) for k,v in crossing.items()})

    # Only the three displayed strategies are simulated; baseline is reused.
    trajectories = {
        stage:{'Control':(age0,y0)}
        for stage in THRESHOLDS
    }

    summary_rows, path_rows = [], []

    for stage,trigger in crossing.items():
        for key in ('Control','PS+20%','Gmeal-10%','Combined+20%'):
            ps_factor,meal_factor = STRATEGIES[key]

            if key=='Control':
                age,y = age0,y0
            else:
                age,y = simulate(
                    args.end_age,args.n_grid,trigger,
                    ps_factor,meal_factor
                )
                trajectories[stage][key] = age,y

            summary_rows.append(
                dict(
                    trigger=stage,
                    strategy=key,
                    intervention_time_years=trigger-FIRST_AGE,
                    G_end_mg_dL=float(y[0,-1]),
                    AUC_above126_mg_dL_year=float(
                        trapezoid(np.maximum(y[0]-126,0),age)
                    )
                )
            )

            for j,a in enumerate(age):
                path_rows.append(
                    (stage,key,float(a-FIRST_AGE),*map(float,y[:,j]))
                )

    pd.DataFrame(summary_rows).to_csv(
        out/'Subject1_intervention_summary.csv',
        index=False
    )

    pd.DataFrame(
        path_rows,
        columns=['trigger','strategy','time_years','G','I','beta','S']
    ).to_csv(
        out/'Subject1_intervention_trajectories.csv',
        index=False
    )

    ends = [
        draw_figure(
            *definition,trajectories,crossing,args.end_age,
            args.stable_tolerance,out
        )
        for definition in FIGURES
    ]

    pd.DataFrame(ends).to_csv(
        out/'Subject1_figure_stability_endpoints.csv',
        index=False
    )

    for entry in ends:
        print(
            f"{entry['figure']}: "
            f"ends at t={entry['figure_end_time_years']:.2f} y"
        )

    print('Saved figures and CSV data in',out.resolve())

if __name__ == '__main__':
    main()

