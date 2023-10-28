import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr

textwidth = 5.905 #inches
def golden_figsize(width):
    return (width, width / 1.618)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 11

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
rc = {
    "mathtext.fontset" : "cm",
    }
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update(rc)

ethGreen = '#627313'
ethBronze = '#8E6713'
ethPurpur = '#A30774'
ethGray = '#6F6F6F'
ethRed = '#B7352D'
ethPetrol = '#007894'
ethPetrolDark = '#00596D'
ethPetrol20 = '#CCE4EA'
ethPetrol40 = '#99CAD5'
ethPurpurDark = '#8C0A59'
ethPurpur40 = '#DC9EC9'

def _figures2_3(fig, dfsurvey, dfstation, params_estim, params_true, 
                res_IM, inf='fixedIM'):
    if inf == 'fixedIM': imstr = 'imB_imS'
    elif inf == 'Bayes': imstr = 'imB_ds_imS'

    subfigs = fig.subfigures(1, 2, wspace=0.03, width_ratios=[1.5, 1.0])
    axsLeft = subfigs[0].subplots(2, 1, 
                                gridspec_kw={'height_ratios': [0.5,1]},
                                sharex=True)
    ax = axsLeft[0]
    for ds in np.arange(3):
        mask = (dfsurvey.DamageState==ds)
        ax.scatter(dfsurvey[mask].x, dfsurvey[mask].DamageState, marker='o', color=ethPetrol,zorder=2,
                    lw = 0.1, s=6, alpha=0.5)
    ax.set_xlim([-20,20])
    ax.set_yticks(np.arange(3))
    ax.set_ylabel('$ds$')
    ax.grid(axis='y', zorder=1, lw=0.1, alpha=0.1)
    ax.text(0, 1.4, 'damage survey data', fontsize=SMALL_SIZE, ha='center')
    ax.set_title('a)', loc='left', fontweight='bold')

    ax = axsLeft[1]
    ax.plot(res_IM.x, res_IM[imstr].loc['q95'], color=ethPetrol40, lw=0.5, ls='-',
            zorder=4)
    ax.plot(res_IM.x, res_IM[imstr].loc['q05'], color=ethPetrol40, lw=0.5, ls='-',
            zorder=4)
    ax.plot(res_IM.x, res_IM[imstr].loc['median'], color=ethPetrol, lw=1.2,zorder=5, label='median', ls='--')

    ax.fill_between(res_IM.x, res_IM[imstr].loc['q95'], res_IM[imstr].loc['q05'], 
                    color=ethPetrol40, zorder=1,alpha=0.5, label='90% CI')
    ax.plot(res_IM.x, res_IM.imB_true, color='black', lw=0.5, zorder=3, label='true')
    ax.set_ylim([0.01,3])
    ax.set_xlim([-20,20])
    ax.scatter(dfstation.x, np.exp(dfstation.obs_logIM.values), 
            color='none', s=25, zorder=6, marker='o', label='station', lw=1.2,
                edgecolor=ethPurpur)
    ax.set_yscale('log')
    ax.set_xlabel('$x$ [km]')
    ax.set_ylabel('$im$ [g]')
    ax.set_ylim([0.01,2])
    ax.set_xlim([-20,20])
    ax.legend(ncol=4, handletextpad=0.5, handlelength=1.5,
            labelspacing=0.2, borderpad=0.25, 
            columnspacing=0.5, loc='lower center', fontsize=SMALL_SIZE)  
    ax.set_title('b)', loc='left', fontweight='bold')

    ax = subfigs[1].subplots(1, 1)
    im = np.linspace(0.001, 1.2, 800)
    logim = np.log(im)
    if inf == 'fixedIM':
        params_estim.plot_frag_funcs(ax, bc='A', im=im, color = ethPetrol, 
                        kwargs={'ls': '--', 'lw': 1.5, 'label': r'fixed $\mathit{IM}$ estimate'})
    elif inf == 'Bayes':
        params_estim.plot_frag_funcs(ax, bc='A', im=im, color = ethPetrol, 
                        kwargsm={'ls': '--', 'lw': 1.5, 'label': r'Bayesian estimate', 'zorder': 6},
                        kwargsCI={'alpha': 0.15, 'zorder': 4})      
    params_true.plot_frag_funcs(ax, bc = 'A', im = im, color = 'black',
                        kwargs={'ls': '-', 'lw':1.0, 'label': 'true'})
    ax.set_xlabel('$im$ [g]')
    ax.set_ylabel(r'$\mathrm{P}(DS\geq ds|im)$')
    ax.set_xlim([0,1.0])
    ax.text(0.45, 0.65, '$ds=1$', fontsize=MEDIUM_SIZE)
    ax.text(0.75, 0.35, '$ds=2$', fontsize=MEDIUM_SIZE)
    ax.set_ylim([-0.02,1.02])
    ax.set_title('c)', loc='left', fontweight='bold')
    ax.legend(handletextpad=0.2, handlelength=1.6,
            labelspacing=0.15, borderpad=0.25, 
            columnspacing=0.5, loc='lower right', fontsize=SMALL_SIZE)


def _figure7(ax, bcs, beta_fixedIM, beta_Bayes):
    x = np.arange(len(bcs))
    vals = {
        r'$\mathrm{fixed}$ $PGA$': tuple(beta_fixedIM['PGA']),
        r'$\mathrm{fixed}$ $SA(0.3s)$': tuple(beta_fixedIM['SAT0_300']),
        r'$\mathrm{Bayesian}$ $PGA$': tuple(np.mean(beta_Bayes['PGA'],axis=1)),
        r'$\mathrm{Bayesian}$ $SA(0.3s)$': tuple(np.mean(beta_Bayes['SAT0_300'],axis=1)),
    }

    qs = {
        r'$\mathrm{Bayesian}$ $PGA$': np.quantile(beta_Bayes['PGA'], [0.15, 0.85], axis=1),
        r'$\mathrm{Bayesian}$ $SA(0.3s)$': np.quantile(beta_Bayes['SAT0_300'], [0.15, 0.85], axis=1),
    }

    width = 0.2
    multiplier = 0

    c = 0
    for attribute, value in vals.items():
        value = np.round(value, 3)
        if attribute == r'$\mathrm{fixed}$ $PGA$': color= ethPurpur; alpha = 0.3
        elif attribute == r'$\mathrm{fixed}$ $SA(0.3s)$': color = ethPurpur; alpha = 0.8
        elif attribute == r'$\mathrm{Bayesian}$ $PGA$': color=ethPetrol; alpha = 0.5
        elif attribute == r'$\mathrm{Bayesian}$ $SA(0.3s)$': color=ethPetrol; alpha = 1.0
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=attribute, color=color, alpha=alpha)
        ax.bar(x + offset, value, width, color='none', edgecolor='black', lw=0.5)
        if attribute.split()[0] == r'$\mathrm{Bayesian}$':
            ax.vlines(x+offset, qs[attribute][0,:], qs[attribute][1,:], color=ethPetrolDark)
            # c += 1
        # ax.bar_label(rects, padding=3)#, fmt = '.2f')
        multiplier += 1
    ax.set_ylabel(r'dispersion $\beta$')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(bcs)
    # ax.legend(loc='upper left', ncol=2)
    ax.legend(ncol=2, handletextpad=0.5, handlelength=1.2,
        labelspacing=0.2, borderpad=0.25, 
        columnspacing=0.5, loc='upper left', fontsize=SMALL_SIZE) 
    ax.set_ylim(0, 1.65)


def _figure10(ax, bcs, beta_fixedIM, beta_Bayes):
    x = np.arange(len(bcs))
    vals = {
        r'fixed $\mathit{IM}$': tuple(beta_fixedIM),
        r'Bayes 1': tuple(np.mean(beta_Bayes[0],axis=1)),
        r'Bayes 2': tuple(np.mean(beta_Bayes[1],axis=1)),
        r'Bayes 3': tuple(np.mean(beta_Bayes[2],axis=1)),
        r'Bayes 4': tuple(np.mean(beta_Bayes[3],axis=1)),
    }

    qs = {
        r'Bayes 1': np.quantile(beta_Bayes[0], [0.05, 0.95], axis=1),
        r'Bayes 2': np.quantile(beta_Bayes[1], [0.05, 0.95], axis=1),
        r'Bayes 3': np.quantile(beta_Bayes[2], [0.05, 0.95], axis=1),
        r'Bayes 4': np.quantile(beta_Bayes[3], [0.05, 0.95], axis=1),
    }

    width = 0.15
    multiplier = 0
    
    c = 0
    for attribute, value in vals.items():
        value = np.round(value, 3)
        if attribute.split()[0] == r'fixed': color= ethPurpur; alpha = 0.3; label = attribute
        elif attribute.split()[0] == 'Bayes': color = ethPetrol; alpha = 0.8; label = None
        if attribute == 'Bayes 1': label='Bayesian (4 random subsamples)' 
        offset = width * multiplier
        rects = ax.bar(x + offset, value, width, label=label, color=color, alpha=alpha)
        ax.bar(x + offset, value, width, color='none', edgecolor='black', lw=0.5)
        if attribute.split()[0] == r'Bayes':
            ax.vlines(x+offset, qs[attribute][0,:], qs[attribute][1,:], color=ethPetrolDark)
            # c += 1
        # ax.bar_label(rects, padding=3)#, fmt = '.2f')
        multiplier += 1
    ax.set_ylabel(r'dispersion $\beta$')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(bcs)
    # ax.legend(loc='upper left', ncol=2)
    ax.legend(ncol=2, handletextpad=0.5, handlelength=1.2,
        labelspacing=0.2, borderpad=0.25, 
        columnspacing=0.5, loc='upper left', fontsize=SMALL_SIZE) 
    ax.set_ylim(0, 1.65)