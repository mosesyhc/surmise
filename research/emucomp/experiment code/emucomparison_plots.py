import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import glob
plt.style.use(['science', 'vibrant', 'grid'])

parent_datadir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\experiment code\emulator_PCGPwM_after_effn_compiled'
output_figdir = r'C:\Users\moses\Desktop\git\surmise\research\emucomp\figs\after_effn'

flist = glob.glob(parent_datadir + r'\*\data\*.json')
d = []
for fname in flist:
    with open(fname, 'r') as f:
        x = json.load(f)
        d.append(json.loads(x))

root_df = pd.DataFrame(d)
root_df['npts'] = root_df.n * root_df.nx * (1 - root_df.failfraction.mean())
root_df.randomfailures = root_df.randomfailures.astype(str)
root_df[['rmse', 'mae', 'medae', 'me', 'crps']] = root_df[['rmse', 'mae', 'medae', 'me', 'crps']].astype(float)

fail_configs = [(True, 'low'), (True, 'high'), (False, 'low'), (False, 'high')] #, (None, 'none')]

markers = ['D', 'v', 'X', 's']
ylabels = {'rmse': 'RMSE',
           # 'mae': 'MAE',
           # 'medae': 'median absolute error',
           # 'me': 'mean error',
           # 'crps': 'CRPS',
           'coverage': r'90\% coverage',
           'avgintwidth': r'90\% interval width',
           # 'intscore': r'interval score',
           'emutime': r'construction time'
           }

funcs = pd.unique(root_df.function)

for fail_random, fail_level in fail_configs:
    df = root_df[(root_df.randomfailures == str(fail_random)) &
                 (root_df.failureslevel == str(fail_level))]
    df.method = df.method.str.replace(r'_', r'-')

    for y, ylabel in ylabels.items():
        fig, ax = plt.subplots(nrows=2, ncols=2,
                               figsize=(8, 6),
                               sharex='all',
                               sharey='all')

        for i, func in enumerate(funcs):
            subdf = df[df.function == func]
            r, c = divmod(i, 2)
            sns.lineplot(x='npts', y=subdf[y],
                         hue='method',
                         style='method',
                         markers=markers,
                         markersize=10,
                         lw=5,
                         alpha=0.7,
                         estimator='mean',
                         ci=None,
                         err_kws={'alpha': 0.0},
                         ax=ax[r][c],
                         data=subdf
                         )
            ax[r][c].set_xlabel('')
            ax[r][c].set_ylabel('')
            ax[r][c].set_yticks([])
            ax[r][c].set_title(func)

        handles, labels = ax[r][c].get_legend_handles_labels()
        for axis in ax.flatten():
            axis.set_xscale('log')
            if y not in ['coverage']:
                axis.set_yscale('log', nonpositive='clip')
            else:
                axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axis.set_ylim((-0.1, 1.1))
            axis.get_legend().remove()

        fig.add_subplot(111, frameon=False)
        fig.legend(handles, labels, loc='lower center', ncol=4)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(ylabel, labelpad=40, fontsize=20)
        plt.xlabel('data size', labelpad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(output_figdir + r'\{:s}_{:s}.png'.format(y, fail_level + '_random' + str(fail_random)))
        # plt.show()
        # break
        plt.close()

        # break
        # plt.show()