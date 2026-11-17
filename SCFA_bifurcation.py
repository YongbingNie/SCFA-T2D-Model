from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parent
GMEAL,GREN,GHEP=1310.,107.,535.
ALPHAS,KI,SG=.0043,.05,8.
SI0,ETA,KS=.635,7.2,100.
SIGMA,AG,GAMMA,KGLP,DI=86400.,20000.,.1,100.,200.
D0,A,GC,RHO,KB,KSC,CG=.0402,.0021,103.,.047,100.,4.,.0037
CASES=[(17.5, (2000.,4400.)), (55.,(2000.,4000.)), (100.,(1800.,3800.))]
COLORS=dict(pos_stable='#2B66BA',pos_unstable='#C95650',boundary_stable='#414349',boundary_unstable='#959BA4')

def h(g,w):
    z=(g-GC)/w
    return D0-A/(1+z**4)

def hp(g,w):
    z=(g-GC)/w
    return 4*A*z**3/(w*(1+z**4)**2)

def ps_of_g(g,w):
    H=h(g,w)
    return KSC*KB*(1+CG*g)*H/(RHO-H)

def dps_dg(g,w):
    H=h(g,w)
    return KSC*KB*(CG*H/(RHO-H)+(1+CG*g)*RHO*hp(g,w)/(RHO-H)**2)

def positive_equilibrium(g,w):
    if g<=0:return None
    ps=ps_of_g(g,w)
    if ps<=0 or not np.isfinite(ps):return None
    s=ps/(KSC*(1+CG*g))
    si=SI0*(1+ETA*s/(KS+s))
    y=GMEAL+GREN-SG*g
    hepat=GHEP/(1+ALPHAS*s)
    if y+hepat<=0:return None
    c=si*g
    disc=(c-KI*y)**2+4*c*KI*(y+hepat)
    I=2*(y+hepat)/(c-KI*y+np.sqrt(disc))
    if not (np.isfinite(I) and I>0):return None
    q=SIGMA*g*g/(AG+g*g)*(1+GAMMA*s/(KGLP+s))
    B=DI*I/q
    qg=SIGMA*2*g*AG/(AG+g*g)**2*(1+GAMMA*s/(KGLP+s))
    qs=SIGMA*g*g/(AG+g*g)*GAMMA*KGLP/(KGLP+s)**2
    sig=SI0*ETA*KS/(KS+s)**2
    z=(g-GC)/w
    fg=-4*A*z**3/(w*(1+z**4)**2)
    fs=RHO*KB/(KB+s)**2
    J=np.array([
        [-SG-si*I,-GHEP*KI/((1+ALPHAS*s)*(1+KI*I)**2)-si*g,0.,-GHEP*ALPHAS/((1+ALPHAS*s)**2*(1+KI*I))-sig*I*g],
        [qg*B,-DI,q,qs*B],
        [B*fg,0.,0.,B*fs],
        [-ps*CG/(1+CG*g)**2,0.,0.,-KSC]
    ])
    eig=np.linalg.eigvals(J)
    return ps,s,I,B,eig

def boundary(ps,w):
    def f(g):
        s=ps/(KSC*(1+CG*g))
        return GMEAL+GREN+GHEP/(1+ALPHAS*s)-SG*g
    g=brentq(f,1e-8,330.)
    s=ps/(KSC*(1+CG*g))
    growth=A/(1+((g-GC)/w)**4)-D0+RHO*s/(KB+s)
    return g,s,growth

def folds(w):
    xs=np.linspace(0.05,225,18000)
    vs=dps_dg(xs,w)
    roots=[]
    for k in np.flatnonzero(vs[:-1]*vs[1:]<0):
        g=brentq(lambda v:dps_dg(v,w),xs[k],xs[k+1])
        e=positive_equilibrium(g,w)
        if e is not None:
            roots.append((float(e[0]),float(g),float(np.max(np.abs(e[-1].real[np.abs(e[-1].real)<1e-6])) if np.any(np.abs(e[-1].real)<1e-6) else np.nan)))
    return sorted(roots)

def contact(w,low,high):
    xs=np.linspace(low,high,700)
    v=np.array([boundary(p,w)[2] for p in xs])
    res=[]
    for k in np.flatnonzero(v[:-1]*v[1:]<0):
        p=brentq(lambda t:boundary(t,w)[2],xs[k],xs[k+1])
        res.append((float(p),float(boundary(p,w)[0])))
    return res

def segments(ax,x,y,stability,palette,lw=2.0):
    # Keep every branch segment in parameterized G-order to preserve folds.
    indexes=np.arange(len(x))
    for status in [True,False]:
        idx=indexes[stability==status]
        if len(idx)<2:continue
        parts=np.split(idx,np.where(np.diff(idx)>1)[0]+1)
        for ids in parts:
            if len(ids)>=2:
                ax.plot(x[ids],y[ids],c=palette[0 if status else 1],lw=lw,
                        ls='-' if status else (0,(5,3)),solid_capstyle='round')

def compute_case(w,window):
    # root family parametrized by positive glucose, not brute-force searches in PS
    ggrid=np.linspace(.07,215.,11500)
    rows=[]
    for g in ggrid:
        e=positive_equilibrium(g,w)
        if e is not None and 0<e[0]<6200:
            rows.append((e[0],g,e[1],e[2],e[3],float(np.max(e[-1].real)),float(np.max(np.abs(e[-1].imag)))))
    df=pd.DataFrame(rows,columns=['P_S','G','S','I','beta','max_Re_eig','max_abs_Im_eig'])
    df['stable']=df.max_Re_eig<-1e-8
    ps=np.linspace(max(window[0]-20,1),window[1]+20,1800)
    boundary_rows=[(p,*boundary(p,w)) for p in ps]
    bdf=pd.DataFrame(boundary_rows,columns=['P_S','G','S','beta_direction_eig'])
    bdf['stable']=bdf.beta_direction_eig<0
    return df,bdf,folds(w),contact(w,max(1,window[0]-600),max(window[1]+900,4600))


def draw(ax, w, xlim, df, bdf, fold_points, contact_points, panel):
    positive=df[(df.P_S>=xlim[0]) & (df.P_S<=xlim[1])]
    bound=bdf[(bdf.P_S>=xlim[0]) & (bdf.P_S<=xlim[1])]
    segments(ax, positive.P_S.to_numpy(), positive.G.to_numpy(),
             positive.stable.to_numpy(),
             (COLORS['pos_stable'],COLORS['pos_unstable']), lw=2.7)
    segments(ax, bound.P_S.to_numpy(), bound.G.to_numpy(),
             bound.stable.to_numpy(),
             (COLORS['boundary_stable'],COLORS['boundary_unstable']), lw=2.15)

    # Offsets are in typographic points and remain fixed under figure resizing.
    for j, (ps, g, _) in enumerate(fold_points):
        if xlim[0] <= ps <= xlim[1]:
            ax.plot(ps, g, marker='o', markersize=6.4, color='#24272A', zorder=8,
                    markeredgecolor='white', markeredgewidth=.55)
            shift = (-12, 13) if j == 0 else (12, -15)
            ha = 'right' if j == 0 else 'left'
            ax.annotate('SN', (ps, g), xytext=shift, textcoords='offset points',
                        ha=ha, va='center', fontsize=15, fontweight='semibold',
                        color='#24272A', zorder=9)
    for ps,g in contact_points:
        if xlim[0] <= ps <= xlim[1]:
            ax.plot(ps, g, marker='s', markersize=7.1, color='#78618D', zorder=8,
                    markeredgecolor='white', markeredgewidth=.6)
            ax.annotate('BP', (ps,g), xytext=(-8,13),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=15, fontweight='medium', color='#33343A', zorder=9)

    # Every horizontal axis ends exactly on a labeled tick.
    xmin,xmax=xlim
    step=400
    xticks=np.arange(xmin, xmax+1, step)
    assert xticks[-1] == xmax
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(5, 220)
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(20, 221, 40))
    ax.set_xlabel(r'$P_S$', fontsize=20, labelpad=8)
    ax.text(.5, -.23, rf'({panel})  $w={w:g}$', transform=ax.transAxes,
            ha='center', va='top', fontsize=19, color='#24272A')
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#D5D8DC', alpha=.35, linewidth=.7)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='both', which='major', direction='out',
                   labelsize=15, length=5.2, width=1.05, pad=5,
                   colors='#35383C')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for name in ['left','bottom']:
        ax.spines[name].set_color('#3A3E43')
        ax.spines[name].set_linewidth(1.05)


def main():
    # Use the real Times New Roman on macOS; avoid font warnings elsewhere.
    try:
        font_manager.findfont('Times New Roman', fallback_to_default=False)
        family = 'Times New Roman'
    except ValueError:
        family = 'Liberation Serif'
    plt.rcParams.update({
        'font.family': family,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    fig,axes=plt.subplots(1,3,figsize=(18.5,6.2),sharey=True)
    for j,(w,xlim) in enumerate(CASES):
        positive,boundary_curve,fold_points,contact_points=compute_case(w,xlim)
        draw(axes[j],w,xlim,positive,boundary_curve,fold_points,contact_points,'abc'[j])
    axes[0].set_ylabel(r'$G$',fontsize=20,labelpad=13)
    fig.subplots_adjust(left=.083,right=.981,top=.955,bottom=.25,wspace=.18)
    for extension,kw in [('.eps',{'dpi':1200}),('.pdf',{}),('.svg',{})]:
        fig.savefig(OUT/('SCFA_bifurcation'+extension),
                    facecolor='white',**kw)
    plt.close(fig)
    print('Saved: SCFA_bifurcation.pdf/.eps/.svg')

if __name__=='__main__':
    main()
