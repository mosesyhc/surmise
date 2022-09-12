import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import seaborn as sns
import itertools
import scipy.stats as sps

def trim_mean(a):
    return sps.trim_mean(a, 0.1)

def q25(a):
    return np.quantile(a, 0.25)

plt.style.use(['science', 'bright', 'grid'])

# parent_datadir = r'./research/emucomp/experiment code/save_MAR_EMGP'
output_figdir = r'./research/emucomp/experiment code/revfigs/MAR_all'
if False:
    flist = glob.glob(parent_datadir + r'\*.json')
    d = []
    for fname in flist:
        with open(fname, 'r') as f:
            x = json.load(f)
            d.append(json.loads(x))


    df = pd.DataFrame(d)
    df.to_json(r'./research/emucomp/experiment code/compiled_dfMAREMGP.json')

root_df = pd.read_csv(r'C:\Users\moses\Desktop\root_df.csv')

root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())
root_df.randomfailures = root_df.randomfailures.astype(str)
root_df[['rmse', 'mae', 'medae', 'me', 'crps']] = root_df[['rmse', 'mae', 'medae', 'me', 'crps']].astype(float)
root_df = root_df.loc[root_df.method != 'PCGP_benchmark']

# fail_configs = [
#     (True, 0.01),
#     (True, 0.05),
#     (True, 0.25),
#     (False, 0.01),
#     (False, 0.05),
#     (False, 0.25),
# ]
fail_configs = [
    ('MAR', 0.01),
    ('MAR', 0.05),
    ('MAR', 0.25)
]

markers = ['D', 'v', 'X', 's', 'o', '^'] #, 'P']
ylabels = {
    'rmse': 'RMSE',
           # 'mae': 'MAE',
           # 'medae': 'median absolute error',
           # 'me': 'mean error',
           # 'crps': 'CRPS',
           'coverage': r'90\% coverage',
           'avgintwidth': r'90\% interval width',
           # 'intscore': r'interval score',
           # 'emutime': r'construction time'
           }

funcs = pd.unique(root_df.function)
funcs[1], funcs[3] = funcs[3], funcs[1]

labels = ['colGP',
             'EMGP',
             'omit',
             'PCGPwM',
             # 'PCGP-benchmark',
             'PCGP-BR',
             'PCGP-KNN']

for fail_random, fail_level in fail_configs:
    df = root_df[(root_df.randomfailures == str(fail_random)) &
                 (root_df.failfraction == fail_level)]
    df.method = df.method.str.replace(r'_', r'-')

    for y, ylabel in ylabels.items():
        std = df[y].std()
        if y == 'avgintwidth':
            # df[y][df[y] > 1e2] = np.nan
            pass
        else:
            df[y][df[y] > 10**6] = np.nan

        if y in ['rmse']:
            est = q25
        elif y in ['coverage']:
            est = 'max'
        else:
            est = trim_mean
        fig, ax = plt.subplots(nrows=2, ncols=2,
                               figsize=(9, 6),
                               sharex='all',
                               )

        for i, func in enumerate(funcs):
            subdf = df[df.function == func]
            r, c = divmod(i, 2)
            sns.lineplot(x='npts', y=subdf[y],
                         hue='method',
                         style='method',
                         markers=markers,
                         markersize=18,
                         lw=4,
                         alpha=0.8,
                         estimator=est,
                         ci=None,
                         # err_kws={'alpha': 0.25},
                         ax=ax[r][c],
                         data=subdf,
                         )
            ax[r][c].set_xlabel('')
            ax[r][c].set_ylabel('')
            ax[r][c].set_yticks([])
            ax[r][c].set_title(func)

        handles, _ = ax[r][c].get_legend_handles_labels()
        for axis in ax.flatten():
            axis.set_xscale('log')
            if y not in ['coverage']:
                axis.set_yscale('log', nonpositive='clip')
            else:
                axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axis.set_ylim((-0.1, 1.1))
            try:
                axis.get_legend().remove()
            except:
                pass

        fig.add_subplot(111, frameon=False)
        fig.legend(handles, labels, loc='lower center',
                   frameon=False, bbox_to_anchor=(0.55, -0.02), ncol=3)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(ylabel, labelpad=40, fontsize=20)
        plt.xlabel('$N$', labelpad=18, fontsize=20)
        plt.tight_layout()
        plt.savefig(output_figdir + r'\{:s}_{:s}.png'.format(y, str(int(fail_level*100)) + '_random' + str(fail_random)))
    #     break
    # break

        plt.close()
        # plt.show()